// Copyright 2022 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Adapted from discord/cros-codecs (`decoder/stateless/h264.rs` and
// `decoder/stateless/h264/vaapi.rs`), itself BSD-3-Clause / Copyright The
// ChromiumOS Authors. See LICENSE.cros-codecs.

//! Thin VA-API H.264 decoder built on the vendored libva binding.
//!
//! The mirror of [`crate::encode`]. It reuses the backend-agnostic bitstream
//! layer ([`crate::codec::h264`]) from discord/cros-codecs (BSD-3-Clause) for
//! SPS/PPS/slice parsing, reference marking, and DPB bumping, then drives libva
//! directly rather than vendoring cros-codecs's generic multi-backend decoder
//! framework. The per-picture and per-slice VA buffer population is ported from
//! cros-codecs's `decoder/stateless/h264/vaapi.rs`.
//!
//! Feed [`Decoder::decode`] one Annex-B access unit at a time. The picture it
//! carries is submitted and completed before the call returns, rather than when
//! the next unit's first slice arrives, so the hardware is never left holding a
//! half-submitted picture between calls. Output still trails, and by more than
//! the stream's reorder depth: C.4.5.3 bumps the DPB only when a new picture
//! needs the slot, so the delay follows the sequence's reference and reorder
//! limits instead. A stream coded with three reference frames trails by three
//! pictures whether or not it uses B-frames. [`Frame`] therefore carries the
//! timestamp of the access unit it was coded in rather than the one last pushed,
//! and [`Decoder::flush`] releases what is left at the end of a stream, which is
//! the whole tail rather than a single picture.
//!
//! Progressive 8-bit 4:2:0 only: a sequence that is interlaced, deeper than 8
//! bits, or not 4:2:0 is rejected rather than decoded wrongly. That covers every
//! stream this crate's encoder, WebRTC, or a browser's `VideoEncoder` produces.

use std::cell::{Cell, RefCell};
use std::io::Cursor;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context as _};

use crate::codec::h264::dpb::{Dpb, DpbEntry, DpbPicRefList, MmcoError, ReferencePicLists};
use crate::codec::h264::parser::{
	Level, MaxLongTermFrameIdx, Nalu, NaluType, Parser, Pps, Profile, RefPicListModification, Slice, SliceHeader,
	SliceType, Sps,
};
use crate::codec::h264::picture::{Field, IsIdr, PictureData, Reference};
use crate::{
	bindings, Buffer, BufferType, Config as VaConfig, Context, Display, DrmPrimeSurfaceDescriptor, H264PicFields,
	H264SeqFields, IQMatrix, IQMatrixBufferH264, Image, Picture, PictureH264, PictureParameter,
	PictureParameterBufferH264, SliceParameter, SliceParameterBufferH264, Surface, UsageHint, VAConfigAttrib,
	VAConfigAttribType, VAEntrypoint, VAProfile, VA_FOURCC_NV12, VA_INVALID_ID, VA_PICTURE_H264_INVALID,
	VA_PICTURE_H264_LONG_TERM_REFERENCE, VA_PICTURE_H264_SHORT_TERM_REFERENCE, VA_RT_FORMAT_YUV420,
	VA_SLICE_DATA_FLAG_ALL,
};

/// H.264 decoder configuration.
#[derive(Clone, Debug)]
pub struct Config {
	/// DRM render node to open (e.g. `/dev/dri/renderD128`).
	pub device: PathBuf,
}

impl Config {
	/// Returns a configuration pointing at the default render node.
	pub fn new() -> Self {
		Self {
			device: PathBuf::from("/dev/dri/renderD128"),
		}
	}
}

impl Default for Config {
	fn default() -> Self {
		Self::new()
	}
}

/// One decoded picture, downloaded to client memory as tightly packed NV12.
#[derive(Clone)]
pub struct Frame {
	/// Timestamp of the access unit this picture was coded in, as handed to
	/// [`Decoder::decode`].
	pub timestamp: u64,
	/// Visible width in pixels, i.e. after the SPS cropping rectangle.
	pub width: u32,
	/// Visible height in pixels.
	pub height: u32,
	/// NV12 planes with no row padding: `width * height` luma bytes followed by
	/// `width * height / 2` interleaved chroma bytes.
	pub data: Vec<u8>,
}

/// One decoded picture, left in the surface the hardware wrote it into.
///
/// The GPU counterpart of [`Frame`]: same picture, same timestamp, but a DRM
/// PRIME descriptor for the surface rather than a copy of its pixels. The
/// descriptor owns the exported file descriptors and keeps the allocation alive
/// on its own, so the surface behind it is already gone by the time this is
/// handed over.
///
/// NV12, and one object holding both planes, which is what the Intel and AMD
/// drivers export. Read the modifier off the object before assuming anything
/// about the layout: a decode target is tiled, and the plane pitches carry
/// padding the visible size does not imply.
pub struct ExportedFrame {
	/// Timestamp of the access unit this picture was coded in, as handed to
	/// [`Decoder::decode_exported`].
	pub timestamp: u64,
	/// Visible width in pixels, i.e. after the SPS cropping rectangle.
	pub width: u32,
	/// Visible height in pixels.
	pub height: u32,
	/// The surface's export: format, modifier, and the offset and pitch of each
	/// plane.
	pub descriptor: DrmPrimeSurfaceDescriptor,
}

impl std::fmt::Debug for ExportedFrame {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ExportedFrame")
			.field("timestamp", &self.timestamp)
			.field("width", &self.width)
			.field("height", &self.height)
			.field("fourcc", &self.descriptor.fourcc)
			.finish()
	}
}

impl std::fmt::Debug for Frame {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Frame")
			.field("timestamp", &self.timestamp)
			.field("width", &self.width)
			.field("height", &self.height)
			.field("bytes", &self.data.len())
			.finish()
	}
}

/// A VA-API H.264 decoder. Built once, fed Annex-B access units, emits NV12.
pub struct Decoder {
	parser: Parser,
	dpb: Dpb<Handle>,
	/// Cached variables from the previous reference picture.
	prev_ref_pic_info: PrevReferencePicInfo,
	/// Cached variables from the previous picture.
	prev_pic_info: PrevPicInfo,
	max_long_term_frame_idx: MaxLongTermFrameIdx,
	/// VA state for the active sequence, rebuilt when the SPS changes something
	/// the config, context, or surface pool depends on.
	sequence: Option<Sequence>,
	/// Pictures bumped out of the DPB, in output order, awaiting download.
	ready: Vec<Handle>,
	/// Drop order: everything above borrows the display.
	display: Arc<Display>,
}

impl Decoder {
	/// Opens the render node and checks that the driver exposes an H.264 decode
	/// entrypoint.
	///
	/// The VA config, context, and surface pool are built later, from the first
	/// SPS, since their profile and size come from the stream. This still fails
	/// early enough for a caller to fall back to another decoder when libva is
	/// missing, the node cannot be opened, or the driver decodes no H.264.
	pub fn new(config: Config) -> anyhow::Result<Self> {
		let display = Display::open_drm_display(&config.device)
			.map_err(|e| anyhow!("open DRM display {:?}: {e:?}", config.device))?;
		probe_decode_entrypoint(&display)?;

		log::info!("opened VA-API H.264 decoder on {:?}", config.device);
		Ok(Self {
			parser: Parser::default(),
			dpb: Dpb::default(),
			prev_ref_pic_info: PrevReferencePicInfo::default(),
			prev_pic_info: PrevPicInfo::default(),
			max_long_term_frame_idx: MaxLongTermFrameIdx::default(),
			sequence: None,
			ready: Vec::new(),
			display,
		})
	}

	/// Decodes one Annex-B access unit, returning the pictures that came due, in
	/// output order.
	///
	/// `timestamp` is opaque and travels with the picture coded in this access
	/// unit, so it survives DPB reordering. Parameter sets are read from the
	/// stream, so an access unit carrying only an SPS and a PPS decodes to no
	/// frames.
	pub fn decode(&mut self, access_unit: &[u8], timestamp: u64) -> anyhow::Result<Vec<Frame>> {
		self.submit(access_unit, timestamp)?;
		self.take_frames()
	}

	/// Decodes one Annex-B access unit, leaving the pictures that came due on
	/// the GPU.
	///
	/// The same decode as [`Decoder::decode`], differing only in what the caller
	/// gets back: a DRM PRIME descriptor for each picture's surface instead of a
	/// copy of its pixels. For a consumer that draws on the GPU that is the
	/// whole picture never touching system memory.
	///
	/// Exporting a surface retires it from the recycling pool, since the next
	/// picture would otherwise be decoded over pixels the consumer still holds a
	/// descriptor for. So this trades a surface allocation per picture for the
	/// download, which is the right way round for anything that would only have
	/// uploaded the pixels again.
	pub fn decode_exported(&mut self, access_unit: &[u8], timestamp: u64) -> anyhow::Result<Vec<ExportedFrame>> {
		self.submit(access_unit, timestamp)?;
		self.take_exported()
	}

	/// Decodes one access unit, leaving the pictures that came due in the ready
	/// queue for whichever of [`Decoder::take_frames`] or
	/// [`Decoder::take_exported`] the caller asked for.
	fn submit(&mut self, access_unit: &[u8], timestamp: u64) -> anyhow::Result<()> {
		let mut cursor = Cursor::new(access_unit);
		let mut current: Option<CurrentPicture> = None;

		// `Nalu::next` reports the end of the buffer as an error, so a truncated
		// unit is indistinguishable from a complete one and simply stops us.
		while let Ok(nalu) = Nalu::next(&mut cursor) {
			match nalu.header.type_ {
				NaluType::Sps => {
					self.parser.parse_sps(&nalu).map_err(|e| anyhow!("parse SPS: {e}"))?;
				}
				NaluType::Pps => {
					self.parser.parse_pps(&nalu).map_err(|e| anyhow!("parse PPS: {e}"))?;
				}
				NaluType::Slice
				| NaluType::SliceDpa
				| NaluType::SliceDpb
				| NaluType::SliceDpc
				| NaluType::SliceIdr
				| NaluType::SliceExt => {
					let slice = self
						.parser
						.parse_slice_header(nalu)
						.map_err(|e| anyhow!("parse slice header: {e}"))?;

					let mut picture = match current.take() {
						// An access unit normally holds one picture, but nothing
						// forces that: `first_mb_in_slice == 0` starts a new one,
						// so the picture in flight is done.
						Some(picture) if slice.header.first_mb_in_slice == 0 => {
							self.finish_picture(picture)?;
							self.begin_picture(&slice, timestamp)?
						}
						Some(picture) => picture,
						None => self.begin_picture(&slice, timestamp)?,
					};

					self.decode_slice(&mut picture, &slice)?;
					current = Some(picture);
				}
				other => log::trace!("ignoring NAL unit of type {other:?}"),
			}
		}

		// Finishing here rather than on the next unit's first slice keeps the
		// hardware from holding a half-submitted picture between calls.
		if let Some(picture) = current.take() {
			self.finish_picture(picture)?;
		}

		Ok(())
	}

	/// Returns every picture still held in the DPB, in output order, and resets
	/// the decoder to await a new IDR.
	pub fn flush(&mut self) -> anyhow::Result<Vec<Frame>> {
		self.reset();
		self.take_frames()
	}

	/// [`Decoder::flush`] for a caller taking its pictures on the GPU.
	pub fn flush_exported(&mut self) -> anyhow::Result<Vec<ExportedFrame>> {
		self.reset();
		self.take_exported()
	}

	/// Moves the DPB to the ready queue and awaits a new IDR.
	fn reset(&mut self) {
		self.drain();
		self.prev_ref_pic_info = PrevReferencePicInfo::default();
		self.prev_pic_info = PrevPicInfo::default();
		self.max_long_term_frame_idx = MaxLongTermFrameIdx::default();
	}

	/// Downloads every picture waiting in the ready queue.
	fn take_frames(&mut self) -> anyhow::Result<Vec<Frame>> {
		std::mem::take(&mut self.ready).iter().map(download).collect()
	}

	/// Exports every picture waiting in the ready queue, leaving the pixels
	/// where the hardware wrote them.
	fn take_exported(&mut self) -> anyhow::Result<Vec<ExportedFrame>> {
		std::mem::take(&mut self.ready).iter().map(export).collect()
	}

	/// Moves everything left in the DPB to the ready queue.
	fn drain(&mut self) {
		self.ready.extend(self.dpb.drain().into_iter().flatten());
	}

	/// Applies `sps` to the DPB limits, rebuilding the VA config, context, and
	/// surface pool if it changes anything they depend on.
	fn apply_sps(&mut self, sps: &Sps) -> anyhow::Result<()> {
		// C.4.5.3: the DPB limits follow the active SPS whether or not the VA
		// state has to be rebuilt.
		let max_dpb_frames = sps.max_dpb_frames();
		let max_num_order_frames = sps.max_num_order_frames() as usize;
		let max_num_reorder_frames = if max_num_order_frames > max_dpb_frames {
			0
		} else {
			max_num_order_frames
		};
		self.dpb.set_limits(max_dpb_frames, max_num_reorder_frames);

		let info = SequenceInfo::new(sps);
		if self.sequence.as_ref().map(|sequence| &sequence.info) == Some(&info) {
			return Ok(());
		}

		// Surfaces are about to change shape, so let everything decoded against
		// the old sequence out first. Those handles keep their own surfaces (and
		// through them the old pool) alive until they are downloaded.
		self.drain();

		self.sequence = Some(Sequence::new(&self.display, sps, info)?);
		Ok(())
	}

	/// Starts a picture: computes its order count, updates the DPB's picture
	/// numbers, and hands the driver the picture parameters.
	fn begin_picture(&mut self, slice: &Slice, timestamp: u64) -> anyhow::Result<CurrentPicture> {
		let pps = Rc::clone(
			self.parser
				.get_pps(slice.header.pic_parameter_set_id)
				.context("slice refers to an unknown PPS")?,
		);
		let sps = Rc::clone(&pps.sps);
		self.apply_sps(&sps)?;

		if slice.nalu.header.idr_pic_flag {
			self.prev_ref_pic_info.frame_num = 0;
		}

		let frame_num = u32::from(slice.header.frame_num);
		if frame_num != self.prev_ref_pic_info.frame_num
			&& frame_num != (self.prev_ref_pic_info.frame_num + 1) % sps.max_frame_num()
		{
			self.handle_frame_num_gap(&sps, frame_num, timestamp)?;
		}

		let pic = self.init_current_pic(slice, &sps, timestamp)?;
		let ref_pic_lists = self.dpb.build_ref_pic_lists(&pic);

		// `apply_sps` above either kept the sequence or built a new one.
		let sequence = self.sequence.as_ref().expect("sequence built by apply_sps");
		let context = Rc::clone(&sequence.context);
		let handle = Handle {
			surface: Rc::new(sequence.pool.alloc()?),
			timestamp,
			width: sequence.width,
			height: sequence.height,
		};

		let mut picture = Picture::new(timestamp, Rc::clone(&context), handle.clone());
		let pic_param = build_pic_param(&slice.header, &pic, handle.surface.id(), &self.dpb, &sps, &pps);
		picture.add_buffer(create_buffer(&context, pic_param)?);
		picture.add_buffer(create_buffer(&context, build_iq_matrix(&pps))?);

		log::trace!("starting picture POC {}", pic.pic_order_cnt);
		Ok(CurrentPicture {
			pic,
			pps,
			handle,
			picture,
			ref_pic_lists,
		})
	}

	/// Hands the driver one slice: its parameters, its reference lists, and the
	/// slice NAL itself.
	fn decode_slice(&mut self, current: &mut CurrentPicture, slice: &Slice) -> anyhow::Result<()> {
		// A slice may refer to a different PPS than the one that started the
		// picture, as long as it names the same SPS.
		let pps = Rc::clone(
			self.parser
				.get_pps(slice.header.pic_parameter_set_id)
				.context("slice refers to an unknown PPS")?,
		);
		if SequenceInfo::new(&pps.sps) != SequenceInfo::new(&current.pps.sps) {
			bail!("invalid stream: the sequence changed between slices of one picture");
		}
		current.pps = Rc::clone(&pps);

		let (list0, list1) = self.create_ref_pic_lists(&current.pic, &slice.header, &current.ref_pic_lists)?;
		let context = &self.sequence.as_ref().expect("sequence built by apply_sps").context;

		let slice_param = build_slice_param(&slice.header, slice.nalu.size, &list0, &list1, &pps.sps, &pps);
		current.picture.add_buffer(create_buffer(context, slice_param)?);
		current.picture.add_buffer(create_buffer(
			context,
			BufferType::SliceData(slice.nalu.as_ref().to_vec()),
		)?);

		Ok(())
	}

	/// Submits the picture to the hardware, then runs the reference marking and
	/// bumping the specification calls for once a picture is decoded.
	fn finish_picture(&mut self, current: CurrentPicture) -> anyhow::Result<()> {
		let CurrentPicture {
			mut pic,
			pps,
			handle,
			picture,
			..
		} = current;

		let picture = picture.begin::<()>().map_err(|e| anyhow!("picture begin: {e:?}"))?;
		let picture = picture.render().map_err(|e| anyhow!("picture render: {e:?}"))?;
		let picture = picture.end().map_err(|e| anyhow!("picture end: {e:?}"))?;
		// Sync before the surface enters the DPB: it is about to be read as a
		// reference by later pictures, and the caller may never ask for its
		// pixels. Dropping the synced picture frees the VA buffers with it.
		drop(picture.sync::<()>().map_err(|(e, _)| anyhow!("picture sync: {e:?}"))?);
		log::trace!("decoded picture POC {}", pic.pic_order_cnt);

		if pic.is_ref() {
			self.reference_pic_marking(&mut pic, &pps.sps)?;
			self.prev_ref_pic_info.fill(&pic);
		}
		self.prev_pic_info.fill(&pic);

		// C.4.5.3 clause 3: a picture with memory_management_control_operation
		// equal to 5 empties the DPB.
		if pic.has_mmco_5 {
			self.drain();
		}

		// C.4.5.3 clauses 1, 4, 5, and 6.
		self.ready.extend(self.dpb.bump_as_needed(&pic).into_iter().flatten());

		// C.4.5.1: a reference picture is bumped until a frame buffer is free,
		// then stored. C.4.5.2: a non-reference picture is stored if a buffer is
		// free after bumping, and output directly otherwise.
		if pic.is_ref() || self.dpb.has_empty_frame_buffer() {
			self.dpb
				.store_picture(pic.into_rc(), Some(handle))
				.map_err(|e| anyhow!("store picture in DPB: {e}"))?;
		} else {
			self.ready.push(handle);
		}

		Ok(())
	}

	/// Builds the picture for `slice` and makes room for it in the DPB.
	fn init_current_pic(&mut self, slice: &Slice, sps: &Sps, timestamp: u64) -> anyhow::Result<PictureData> {
		// Progressive only, so a picture is never the second field of a frame.
		let mut pic = PictureData::new_from_slice(slice, sps, timestamp, None);
		self.compute_pic_order_count(&mut pic, sps)?;

		if matches!(pic.is_idr, IsIdr::Yes { .. }) {
			// C.4.5.3 clause 2: an IDR bumps the whole DPB, unless
			// no_output_of_prior_pics_flag says to discard it instead (C.4.4).
			if pic.ref_pic_marking.no_output_of_prior_pics_flag {
				self.dpb.clear();
			} else {
				self.drain();
			}
		}

		self.dpb
			.update_pic_nums(u32::from(slice.header.frame_num), sps.max_frame_num(), &pic);

		Ok(pic)
	}

	/// 8.2.5.2: invents the pictures for the frame numbers a lossy stream skipped,
	/// so reference marking keeps working.
	fn handle_frame_num_gap(&mut self, sps: &Sps, frame_num: u32, timestamp: u64) -> anyhow::Result<()> {
		if self.dpb.is_empty() {
			return Ok(());
		}

		if !sps.gaps_in_frame_num_value_allowed_flag {
			bail!("invalid frame_num {frame_num}: assuming unintentional loss of pictures");
		}
		log::debug!("frame_num gap up to {frame_num}");

		let max_frame_num = sps.max_frame_num();
		let mut unused = (self.prev_ref_pic_info.frame_num + 1) % max_frame_num;
		while unused != frame_num {
			let mut pic = PictureData::new_non_existing(unused, timestamp);
			self.compute_pic_order_count(&mut pic, sps)?;

			self.dpb.update_pic_nums(unused, max_frame_num, &pic);
			self.dpb.sliding_window_marking(&mut pic, sps);
			self.ready.extend(self.dpb.bump_as_needed(&pic).into_iter().flatten());
			self.dpb
				.store_picture(pic.into_rc(), None)
				.map_err(|e| anyhow!("store non-existing picture in DPB: {e}"))?;

			unused = (unused + 1) % max_frame_num;
		}

		Ok(())
	}

	/// 8.2.1: derives the picture order count, which drives output ordering.
	fn compute_pic_order_count(&mut self, pic: &mut PictureData, sps: &Sps) -> anyhow::Result<()> {
		match pic.pic_order_cnt_type {
			// 8.2.1.1
			0 => {
				let (prev_pic_order_cnt_msb, prev_pic_order_cnt_lsb) = if matches!(pic.is_idr, IsIdr::Yes { .. }) {
					(0, 0)
				} else if self.prev_ref_pic_info.has_mmco_5 {
					if matches!(self.prev_ref_pic_info.field, Field::Bottom) {
						(0, 0)
					} else {
						(0, self.prev_ref_pic_info.top_field_order_cnt)
					}
				} else {
					(
						self.prev_ref_pic_info.pic_order_cnt_msb,
						self.prev_ref_pic_info.pic_order_cnt_lsb,
					)
				};

				let max_pic_order_cnt_lsb = 1 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);

				pic.pic_order_cnt_msb = if (pic.pic_order_cnt_lsb < self.prev_ref_pic_info.pic_order_cnt_lsb)
					&& (prev_pic_order_cnt_lsb - pic.pic_order_cnt_lsb >= max_pic_order_cnt_lsb / 2)
				{
					prev_pic_order_cnt_msb + max_pic_order_cnt_lsb
				} else if (pic.pic_order_cnt_lsb > prev_pic_order_cnt_lsb)
					&& (pic.pic_order_cnt_lsb - prev_pic_order_cnt_lsb > max_pic_order_cnt_lsb / 2)
				{
					prev_pic_order_cnt_msb - max_pic_order_cnt_lsb
				} else {
					prev_pic_order_cnt_msb
				};

				if !matches!(pic.field, Field::Bottom) {
					pic.top_field_order_cnt = pic.pic_order_cnt_msb + pic.pic_order_cnt_lsb;
				}

				if !matches!(pic.field, Field::Top) {
					pic.bottom_field_order_cnt = if matches!(pic.field, Field::Frame) {
						pic.top_field_order_cnt + pic.delta_pic_order_cnt_bottom
					} else {
						pic.pic_order_cnt_msb + pic.pic_order_cnt_lsb
					};
				}
			}

			// 8.2.1.2
			1 => {
				if self.prev_pic_info.has_mmco_5 {
					self.prev_pic_info.frame_num_offset = 0;
				}

				pic.frame_num_offset = if matches!(pic.is_idr, IsIdr::Yes { .. }) {
					0
				} else if self.prev_pic_info.frame_num > pic.frame_num {
					self.prev_pic_info.frame_num_offset + sps.max_frame_num()
				} else {
					self.prev_pic_info.frame_num_offset
				};

				let mut abs_frame_num = if sps.num_ref_frames_in_pic_order_cnt_cycle != 0 {
					pic.frame_num_offset + pic.frame_num
				} else {
					0
				};

				if pic.nal_ref_idc == 0 && abs_frame_num > 0 {
					abs_frame_num -= 1;
				}

				let mut expected_pic_order_cnt = 0;

				if abs_frame_num > 0 {
					if sps.num_ref_frames_in_pic_order_cnt_cycle == 0 {
						bail!("invalid num_ref_frames_in_pic_order_cnt_cycle");
					}

					let cycles = (abs_frame_num - 1) / u32::from(sps.num_ref_frames_in_pic_order_cnt_cycle);
					expected_pic_order_cnt = cycles as i32 * sps.expected_delta_per_pic_order_cnt_cycle;

					for i in 0..sps.num_ref_frames_in_pic_order_cnt_cycle {
						expected_pic_order_cnt += sps.offset_for_ref_frame[usize::from(i)];
					}
				}

				if pic.nal_ref_idc == 0 {
					expected_pic_order_cnt += sps.offset_for_non_ref_pic;
				}

				match pic.field {
					Field::Frame => {
						pic.top_field_order_cnt = expected_pic_order_cnt + pic.delta_pic_order_cnt0;
						pic.bottom_field_order_cnt =
							pic.top_field_order_cnt + sps.offset_for_top_to_bottom_field + pic.delta_pic_order_cnt1;
					}
					Field::Top => {
						pic.top_field_order_cnt = expected_pic_order_cnt + pic.delta_pic_order_cnt0;
					}
					Field::Bottom => {
						pic.bottom_field_order_cnt =
							expected_pic_order_cnt + sps.offset_for_top_to_bottom_field + pic.delta_pic_order_cnt0;
					}
				}
			}

			// 8.2.1.3
			2 => {
				if self.prev_pic_info.has_mmco_5 {
					self.prev_pic_info.frame_num_offset = 0;
				}

				pic.frame_num_offset = if matches!(pic.is_idr, IsIdr::Yes { .. }) {
					0
				} else if self.prev_pic_info.frame_num > pic.frame_num {
					self.prev_pic_info.frame_num_offset + sps.max_frame_num()
				} else {
					self.prev_pic_info.frame_num_offset
				};

				let pic_order_cnt = if matches!(pic.is_idr, IsIdr::Yes { .. }) {
					0
				} else if pic.nal_ref_idc == 0 {
					2 * (pic.frame_num_offset + pic.frame_num) as i32 - 1
				} else {
					2 * (pic.frame_num_offset + pic.frame_num) as i32
				};

				if matches!(pic.field, Field::Frame | Field::Top) {
					pic.top_field_order_cnt = pic_order_cnt;
				}
				if matches!(pic.field, Field::Frame | Field::Bottom) {
					pic.bottom_field_order_cnt = pic_order_cnt;
				}
			}

			other => bail!("invalid pic_order_cnt_type {other}"),
		}

		pic.pic_order_cnt = match pic.field {
			Field::Frame => std::cmp::min(pic.top_field_order_cnt, pic.bottom_field_order_cnt),
			Field::Top => pic.top_field_order_cnt,
			Field::Bottom => pic.bottom_field_order_cnt,
		};

		Ok(())
	}

	/// 8.2.5.1: marks the decoded picture as used or unused for reference.
	fn reference_pic_marking(&mut self, pic: &mut PictureData, sps: &Sps) -> anyhow::Result<()> {
		if matches!(pic.is_idr, IsIdr::Yes { .. }) {
			self.dpb.mark_all_as_unused_for_ref();

			if pic.ref_pic_marking.long_term_reference_flag {
				pic.set_reference(Reference::LongTerm, false);
				pic.long_term_frame_idx = 0;
				self.max_long_term_frame_idx = MaxLongTermFrameIdx::Idx(0);
			} else {
				pic.set_reference(Reference::ShortTerm, false);
				self.max_long_term_frame_idx = MaxLongTermFrameIdx::NoLongTermFrameIndices;
			}

			return Ok(());
		}

		if pic.ref_pic_marking.adaptive_ref_pic_marking_mode_flag {
			self.handle_memory_management_ops(pic)?;
		} else {
			self.dpb.sliding_window_marking(pic, sps);
		}

		Ok(())
	}

	/// 8.2.5.4: runs the picture's adaptive memory management control operations.
	fn handle_memory_management_ops(&mut self, pic: &mut PictureData) -> Result<(), MmcoError> {
		let markings = pic.ref_pic_marking.clone();

		for marking in &markings.inner {
			match marking.memory_management_control_operation {
				0 => break,
				1 => self.dpb.mmco_op_1(pic, marking)?,
				2 => self.dpb.mmco_op_2(pic, marking)?,
				3 => self.dpb.mmco_op_3(pic, marking)?,
				4 => self.max_long_term_frame_idx = self.dpb.mmco_op_4(marking),
				5 => self.max_long_term_frame_idx = self.dpb.mmco_op_5(pic),
				6 => self.dpb.mmco_op_6(pic, marking),
				other => return Err(MmcoError::UnknownMmco(other)),
			}
		}

		Ok(())
	}

	/// 8.2.4: derives RefPicList0 and RefPicList1 for one slice, from the
	/// per-picture lists the DPB built.
	fn create_ref_pic_lists<'a>(
		&'a self,
		pic: &PictureData,
		hdr: &SliceHeader,
		lists: &ReferencePicLists,
	) -> anyhow::Result<(DpbPicRefList<'a, Handle>, DpbPicRefList<'a, Handle>)> {
		let list0 = match hdr.slice_type {
			SliceType::P | SliceType::Sp => self.modify_ref_pic_list(pic, hdr, false, &lists.ref_pic_list_p0)?,
			SliceType::B => self.modify_ref_pic_list(pic, hdr, false, &lists.ref_pic_list_b0)?,
			_ => Vec::new(),
		};

		let list1 = match hdr.slice_type {
			SliceType::B => self.modify_ref_pic_list(pic, hdr, true, &lists.ref_pic_list_b1)?,
			_ => Vec::new(),
		};

		Ok((list0, list1))
	}

	/// 8.2.4.3: applies the slice header's reference picture list modifications.
	fn modify_ref_pic_list<'a>(
		&'a self,
		pic: &PictureData,
		hdr: &SliceHeader,
		list1: bool,
		indices: &[usize],
	) -> anyhow::Result<DpbPicRefList<'a, Handle>> {
		let (modify, num_ref_idx_active_minus1, modifications) = if list1 {
			(
				hdr.ref_pic_list_modification_flag_l1,
				hdr.num_ref_idx_l1_active_minus1,
				&hdr.ref_pic_list_modification_l1,
			)
		} else {
			(
				hdr.ref_pic_list_modification_flag_l0,
				hdr.num_ref_idx_l0_active_minus1,
				&hdr.ref_pic_list_modification_l0,
			)
		};

		let mut list: DpbPicRefList<'a, Handle> = indices
			.iter()
			.map(|&i| &self.dpb.entries()[i])
			.take(usize::from(num_ref_idx_active_minus1) + 1)
			.collect();

		if !modify {
			return Ok(list);
		}

		let mut pic_num_pred = pic.pic_num;
		let mut ref_idx = 0;

		for modification in modifications {
			match modification.modification_of_pic_nums_idc {
				idc @ (0 | 1) => self.short_term_modification(
					pic,
					&mut list,
					num_ref_idx_active_minus1,
					hdr.max_pic_num as i32,
					idc,
					modification,
					&mut pic_num_pred,
					&mut ref_idx,
				)?,
				2 => self.long_term_modification(&mut list, num_ref_idx_active_minus1, modification, &mut ref_idx)?,
				3 => break,
				other => bail!("unexpected modification_of_pic_nums_idc {other}"),
			}
		}

		Ok(list)
	}

	/// 8.2.4.3.1: modification for short-term reference pictures.
	#[allow(clippy::too_many_arguments)]
	fn short_term_modification<'a>(
		&'a self,
		pic: &PictureData,
		list: &mut DpbPicRefList<'a, Handle>,
		num_ref_idx_active_minus1: u8,
		max_pic_num: i32,
		idc: u8,
		modification: &RefPicListModification,
		pic_num_pred: &mut i32,
		ref_idx: &mut usize,
	) -> anyhow::Result<()> {
		let abs_diff_pic_num = modification.abs_diff_pic_num_minus1 as i32 + 1;

		let pic_num_no_wrap = if idc == 0 {
			if *pic_num_pred - abs_diff_pic_num < 0 {
				*pic_num_pred - abs_diff_pic_num + max_pic_num
			} else {
				*pic_num_pred - abs_diff_pic_num
			}
		} else if *pic_num_pred + abs_diff_pic_num >= max_pic_num {
			*pic_num_pred + abs_diff_pic_num - max_pic_num
		} else {
			*pic_num_pred + abs_diff_pic_num
		};

		*pic_num_pred = pic_num_no_wrap;

		let pic_num = if pic_num_no_wrap > pic.pic_num {
			pic_num_no_wrap - max_pic_num
		} else {
			pic_num_no_wrap
		};

		let entry = self
			.dpb
			.find_short_term_with_pic_num(pic_num)
			.with_context(|| format!("no short-term reference with pic_num {pic_num}"))?;

		Self::insert_modification(list, entry, num_ref_idx_active_minus1, ref_idx, |target| {
			target.pic_num_f(max_pic_num) != pic_num
		});

		Ok(())
	}

	/// 8.2.4.3.2: modification for long-term reference pictures.
	fn long_term_modification<'a>(
		&'a self,
		list: &mut DpbPicRefList<'a, Handle>,
		num_ref_idx_active_minus1: u8,
		modification: &RefPicListModification,
		ref_idx: &mut usize,
	) -> anyhow::Result<()> {
		let long_term_pic_num = modification.long_term_pic_num;
		let max_long_term_frame_idx = self.max_long_term_frame_idx;

		let entry = self
			.dpb
			.find_long_term_with_long_term_pic_num(long_term_pic_num)
			.with_context(|| format!("no long-term reference with long_term_pic_num {long_term_pic_num}"))?;

		Self::insert_modification(list, entry, num_ref_idx_active_minus1, ref_idx, |target| {
			target.long_term_pic_num_f(max_long_term_frame_idx) != long_term_pic_num
		});

		Ok(())
	}

	/// The shared tail of 8.2.4.3.1 and 8.2.4.3.2: insert `entry` at `ref_idx`,
	/// then shuffle out the duplicate `keep` identifies and re-truncate the list.
	fn insert_modification<'a>(
		list: &mut DpbPicRefList<'a, Handle>,
		entry: &'a DpbEntry<Handle>,
		num_ref_idx_active_minus1: u8,
		ref_idx: &mut usize,
		keep: impl Fn(&PictureData) -> bool,
	) {
		// A modification past the end of the list is a stream error; the spec
		// only defines indices within num_ref_idx_lX_active_minus1 + 1.
		let at = (*ref_idx).min(list.len());
		list.insert(at, entry);
		*ref_idx = at + 1;

		let mut next = *ref_idx;
		for current in *ref_idx..=usize::from(num_ref_idx_active_minus1) + 1 {
			if current >= list.len() {
				break;
			}

			if keep(&list[current].pic.borrow()) {
				list[next] = list[current];
				next += 1;
			}
		}

		list.truncate(usize::from(num_ref_idx_active_minus1) + 1);
	}
}

/// The picture currently being assembled, held across the slices that make it up.
struct CurrentPicture {
	pic: PictureData,
	/// PPS active for the last slice seen.
	pps: Rc<Pps>,
	/// Handle to the surface the driver renders into.
	handle: Handle,
	picture: Picture<crate::PictureNew, Handle>,
	/// Per-picture reference lists, modified per slice.
	ref_pic_lists: ReferencePicLists,
}

/// A decoded picture: the surface holding it plus what the caller needs to read
/// it back. Cloned into the DPB, so the surface is not recycled while it is still
/// a reference or still waiting to be output.
#[derive(Clone)]
struct Handle {
	surface: Rc<PooledSurface>,
	timestamp: u64,
	width: u32,
	height: u32,
}

impl std::borrow::Borrow<Surface<()>> for Handle {
	fn borrow(&self) -> &Surface<()> {
		self.surface.get()
	}
}

/// The properties of an SPS that the VA config, context, and surface pool are
/// built from. A change in any of them forces them to be rebuilt.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SequenceInfo {
	coded: (u32, u32),
	profile_idc: u8,
	bit_depth_luma_minus8: u8,
	bit_depth_chroma_minus8: u8,
	chroma_format_idc: u8,
	frame_mbs_only_flag: bool,
	max_dpb_frames: usize,
}

impl SequenceInfo {
	fn new(sps: &Sps) -> Self {
		Self {
			coded: (sps.width(), sps.height()),
			profile_idc: sps.profile_idc,
			bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
			bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
			chroma_format_idc: sps.chroma_format_idc,
			frame_mbs_only_flag: sps.frame_mbs_only_flag,
			max_dpb_frames: sps.max_dpb_frames(),
		}
	}
}

/// The VA state for one coded video sequence.
struct Sequence {
	info: SequenceInfo,
	/// Visible width after cropping, which is what a [`Frame`] reports.
	width: u32,
	/// Visible height after cropping.
	height: u32,
	pool: Rc<SurfacePool>,
	// Drop order: the context must go before the config it was created from.
	context: Rc<Context>,
	_config: VaConfig,
}

impl Sequence {
	fn new(display: &Arc<Display>, sps: &Sps, info: SequenceInfo) -> anyhow::Result<Self> {
		if !sps.frame_mbs_only_flag {
			bail!("interlaced H.264 is not supported");
		}
		if sps.chroma_format_idc != 1 {
			bail!(
				"only 4:2:0 chroma is supported, got chroma_format_idc {}",
				sps.chroma_format_idc
			);
		}
		if sps.bit_depth_luma_minus8 != 0 || sps.bit_depth_chroma_minus8 != 0 {
			bail!(
				"only 8-bit samples are supported, got {}-bit luma and {}-bit chroma",
				sps.bit_depth_luma_minus8 + 8,
				sps.bit_depth_chroma_minus8 + 8
			);
		}

		let visible = sps.visible_rectangle();
		let width = visible.max.x - visible.min.x;
		let height = visible.max.y - visible.min.y;
		if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
			bail!("visible size {width}x{height} is not a non-zero even 4:2:0 size");
		}

		let attrs = vec![VAConfigAttrib {
			type_: VAConfigAttribType::VAConfigAttribRTFormat,
			value: VA_RT_FORMAT_YUV420,
		}];
		let config = display
			.create_config(attrs, va_profile(sps)?, VAEntrypoint::VAEntrypointVLD)
			.map_err(|e| anyhow!("create VA config: {e:?}"))?;

		let (coded_width, coded_height) = info.coded;
		let context = display
			.create_context::<()>(&config, coded_width, coded_height, None, true)
			.map_err(|e| anyhow!("create VA context: {e:?}"))?;

		log::info!(
			"VA-API H.264 sequence: {width}x{height} visible, {coded_width}x{coded_height} coded, \
			 profile_idc {}, DPB {} frames",
			sps.profile_idc,
			info.max_dpb_frames
		);

		Ok(Self {
			info,
			width,
			height,
			pool: Rc::new(SurfacePool {
				display: Arc::clone(display),
				width: coded_width,
				height: coded_height,
				free: RefCell::new(Vec::new()),
			}),
			context,
			_config: config,
		})
	}
}

/// A recycling pool of decode target surfaces.
///
/// Surfaces are allocated on demand and returned when the last handle referring
/// to one drops. The DPB bounds how many can be in flight at once, so the pool
/// stops growing on its own after the first few pictures.
struct SurfacePool {
	display: Arc<Display>,
	width: u32,
	height: u32,
	free: RefCell<Vec<Surface<()>>>,
}

impl SurfacePool {
	fn alloc(self: &Rc<Self>) -> anyhow::Result<PooledSurface> {
		let surface = match self.free.borrow_mut().pop() {
			Some(surface) => surface,
			None => self
				.display
				.create_surfaces::<()>(
					VA_RT_FORMAT_YUV420,
					Some(VA_FOURCC_NV12),
					self.width,
					self.height,
					Some(UsageHint::USAGE_HINT_DECODER),
					vec![()],
				)
				.map_err(|e| anyhow!("create decode surface: {e:?}"))?
				.pop()
				.context("VA-API returned an empty surface list")?,
		};

		Ok(PooledSurface {
			surface: Some(surface),
			pool: Rc::clone(self),
			retired: Cell::new(false),
		})
	}
}

/// A surface on loan from a [`SurfacePool`], returned to it on drop.
struct PooledSurface {
	/// Always `Some` until dropped; the `Option` is only there so [`Drop`] can
	/// move the surface back into the pool.
	surface: Option<Surface<()>>,
	pool: Rc<SurfacePool>,
	/// Set once this surface's allocation has been handed out as a DRM PRIME
	/// descriptor. Recycling it then would have the decoder draw its next
	/// picture over pixels the consumer is still reading.
	retired: Cell<bool>,
}

impl PooledSurface {
	fn get(&self) -> &Surface<()> {
		self.surface.as_ref().expect("surface is taken only on drop")
	}

	fn id(&self) -> bindings::VASurfaceID {
		self.get().id()
	}

	/// Keeps this surface out of the pool when the last handle to it drops.
	fn retire(&self) {
		self.retired.set(true);
	}
}

impl Drop for PooledSurface {
	fn drop(&mut self) {
		if let Some(surface) = self.surface.take() {
			// A retired surface is destroyed here, which releases libva's
			// reference on the underlying buffer. The exported descriptor holds
			// one of its own, so the pixels outlive this.
			if !self.retired.get() {
				self.pool.free.borrow_mut().push(surface);
			}
		}
	}
}

/// Exports a decoded surface as a DRM PRIME descriptor, copying nothing.
///
/// The surface is retired from the pool on the way out: the descriptor refers to
/// the same allocation, so recycling it would have a later picture overwrite one
/// the caller still holds.
fn export(handle: &Handle) -> anyhow::Result<ExportedFrame> {
	let surface = handle.surface.get();
	surface.sync().map_err(|e| anyhow!("surface sync: {e:?}"))?;

	let descriptor = surface
		.export_prime()
		.map_err(|e| anyhow!("export the decode surface: {e:?}"))?;
	handle.surface.retire();

	Ok(ExportedFrame {
		timestamp: handle.timestamp,
		width: handle.width,
		height: handle.height,
		descriptor,
	})
}

/// Reads a decoded surface back to client memory as tightly packed NV12.
fn download(handle: &Handle) -> anyhow::Result<Frame> {
	let surface = handle.surface.get();
	surface.sync().map_err(|e| anyhow!("surface sync: {e:?}"))?;

	let visible = (handle.width, handle.height);

	// vaDeriveImage exposes the surface's own buffer, so the copy below is the
	// only one. Not every driver can derive every surface; fall back to
	// vaCreateImage + vaGetImage, which always works but copies inside the driver
	// first.
	match Image::derive_from(surface, visible) {
		Ok(image) if image.image().format.fourcc == VA_FOURCC_NV12 => return pack_nv12(&image, handle),
		Ok(_) => log::debug!("derived image is not NV12, falling back to vaGetImage"),
		Err(e) => log::debug!("vaDeriveImage failed ({e:?}), falling back to vaGetImage"),
	}

	let format = surface
		.display()
		.query_image_formats()
		.map_err(|e| anyhow!("query image formats: {e:?}"))?
		.into_iter()
		.find(|format| format.fourcc == VA_FOURCC_NV12)
		.context("driver has no NV12 image format")?;
	let image = Image::create_from(surface, format, surface.size(), visible)
		.map_err(|e| anyhow!("vaGetImage into an NV12 image: {e:?}"))?;

	pack_nv12(&image, handle)
}

/// Copies a mapped NV12 image into a tightly packed buffer, dropping the row
/// padding the driver's pitch may carry.
fn pack_nv12(image: &Image<'_>, handle: &Handle) -> anyhow::Result<Frame> {
	let va_image = image.image();
	let src: &[u8] = image.as_ref();
	let (width, height) = (handle.width as usize, handle.height as usize);
	let chroma_rows = height / 2;

	let mut data = vec![0u8; width * (height + chroma_rows)];
	let (luma, chroma) = data.split_at_mut(width * height);
	copy_plane(
		luma,
		src,
		va_image.offsets[0] as usize,
		va_image.pitches[0] as usize,
		width,
		height,
	)?;
	copy_plane(
		chroma,
		src,
		va_image.offsets[1] as usize,
		va_image.pitches[1] as usize,
		width,
		chroma_rows,
	)?;

	Ok(Frame {
		timestamp: handle.timestamp,
		width: handle.width,
		height: handle.height,
		data,
	})
}

/// Copies `rows` rows of `width` bytes out of a strided plane.
fn copy_plane(
	dst: &mut [u8],
	src: &[u8],
	offset: usize,
	pitch: usize,
	width: usize,
	rows: usize,
) -> anyhow::Result<()> {
	if rows == 0 {
		return Ok(());
	}
	let end = offset + pitch * (rows - 1) + width;
	if pitch < width || src.len() < end {
		bail!("VA image plane (offset {offset}, pitch {pitch}) does not hold {rows} rows of {width} bytes");
	}

	for row in 0..rows {
		let from = offset + row * pitch;
		dst[row * width..][..width].copy_from_slice(&src[from..from + width]);
	}

	Ok(())
}

/// Fails unless the driver can decode H.264 at some profile we can drive.
fn probe_decode_entrypoint(display: &Display) -> anyhow::Result<()> {
	let profiles = display
		.query_config_profiles()
		.map_err(|e| anyhow!("query VA profiles: {e:?}"))?;

	let supported = profiles
		.into_iter()
		.filter(|profile| {
			matches!(
				*profile,
				VAProfile::VAProfileH264ConstrainedBaseline
					| VAProfile::VAProfileH264Main
					| VAProfile::VAProfileH264High
			)
		})
		.any(|profile| {
			display
				.query_config_entrypoints(profile)
				.is_ok_and(|entrypoints| entrypoints.contains(&VAEntrypoint::VAEntrypointVLD))
		});

	if !supported {
		bail!("driver exposes no H.264 decode entrypoint");
	}

	Ok(())
}

/// Maps the SPS profile onto the VA profile to open the config with.
fn va_profile(sps: &Sps) -> anyhow::Result<VAProfile::Type> {
	let profile = Profile::try_from(sps.profile_idc).map_err(|e| anyhow!("{e}"))?;

	Ok(match profile {
		// VA-API has no plain Baseline profile, and the picture parameter buffer
		// has no way to describe the FMO and ASO tools that are all that separate
		// it from Constrained Baseline (num_slice_groups_minus1 is pinned to 0
		// below), so both map here. A stream that really uses them decodes wrong
		// rather than not at all, which no encoder in this decade emits.
		Profile::Baseline => VAProfile::VAProfileH264ConstrainedBaseline,
		Profile::Main | Profile::Extended => VAProfile::VAProfileH264Main,
		// The bit depth and chroma format are checked separately, so a High10 or
		// High422 SPS only reaches here when its samples are 8-bit 4:2:0.
		Profile::High | Profile::High10 | Profile::High422P => VAProfile::VAProfileH264High,
	})
}

fn create_buffer(context: &Rc<Context>, buffer: BufferType) -> anyhow::Result<Buffer> {
	context
		.create_buffer(buffer)
		.map_err(|e| anyhow!("create VA buffer: {e:?}"))
}

/// Cached variables from the previous reference picture, needed by 8.2.1.
#[derive(Debug)]
struct PrevReferencePicInfo {
	frame_num: u32,
	has_mmco_5: bool,
	top_field_order_cnt: i32,
	pic_order_cnt_msb: i32,
	pic_order_cnt_lsb: i32,
	field: Field,
}

impl Default for PrevReferencePicInfo {
	fn default() -> Self {
		Self {
			frame_num: 0,
			has_mmco_5: false,
			top_field_order_cnt: 0,
			pic_order_cnt_msb: 0,
			pic_order_cnt_lsb: 0,
			field: Field::Frame,
		}
	}
}

impl PrevReferencePicInfo {
	fn fill(&mut self, pic: &PictureData) {
		self.has_mmco_5 = pic.has_mmco_5;
		self.top_field_order_cnt = pic.top_field_order_cnt;
		self.pic_order_cnt_msb = pic.pic_order_cnt_msb;
		self.pic_order_cnt_lsb = pic.pic_order_cnt_lsb;
		self.field = pic.field;
		self.frame_num = pic.frame_num;
	}
}

/// Cached variables from the previous picture, needed by 8.2.1.
#[derive(Debug, Default)]
struct PrevPicInfo {
	frame_num: u32,
	frame_num_offset: u32,
	has_mmco_5: bool,
}

impl PrevPicInfo {
	fn fill(&mut self, pic: &PictureData) {
		self.frame_num = pic.frame_num;
		self.has_mmco_5 = pic.has_mmco_5;
		self.frame_num_offset = pic.frame_num_offset;
	}
}

/// The surface a DPB entry decoded into, or `VA_INVALID_ID` for the non-existing
/// pictures invented for a frame_num gap.
fn va_surface_id(handle: &Option<Handle>) -> bindings::VASurfaceID {
	match handle {
		Some(handle) => handle.surface.id(),
		None => VA_INVALID_ID,
	}
}

/// Ported from cros-codecs `fill_va_h264_pic` (BSD-3-Clause). Progressive only,
/// so the field cases collapse to the frame one.
fn fill_va_h264_pic(pic: &PictureData, surface_id: bindings::VASurfaceID) -> PictureH264 {
	let mut flags = 0;
	let frame_idx = match pic.reference() {
		Reference::LongTerm => {
			flags |= VA_PICTURE_H264_LONG_TERM_REFERENCE;
			pic.long_term_frame_idx
		}
		Reference::ShortTerm => {
			flags |= VA_PICTURE_H264_SHORT_TERM_REFERENCE;
			pic.frame_num
		}
		Reference::None => pic.frame_num,
	};

	PictureH264::new(
		surface_id,
		frame_idx,
		flags,
		pic.top_field_order_cnt,
		pic.bottom_field_order_cnt,
	)
}

/// Builds the picture used to fill the array slots there is no reference for.
fn build_invalid_va_h264_pic() -> PictureH264 {
	PictureH264::new(VA_INVALID_ID, 0, VA_PICTURE_H264_INVALID, 0, 0)
}

/// Ported from cros-codecs `build_iq_matrix` (BSD-3-Clause). VA-API wants the
/// scaling lists in raster order, the PPS carries them zig-zag.
fn build_iq_matrix(pps: &Pps) -> BufferType {
	const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
	#[rustfmt::skip]
	const ZIGZAG_8X8: [usize; 64] = [
		0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
		20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
		58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
	];

	let mut scaling_list4x4 = [[0u8; 16]; 6];
	for (list, src) in scaling_list4x4.iter_mut().zip(pps.scaling_lists_4x4.iter()) {
		for (i, &value) in src.iter().enumerate() {
			list[ZIGZAG_4X4[i]] = value;
		}
	}

	let mut scaling_list8x8 = [[0u8; 64]; 2];
	for (list, src) in scaling_list8x8.iter_mut().zip(pps.scaling_lists_8x8.iter()) {
		for (i, &value) in src.iter().enumerate() {
			list[ZIGZAG_8X8[i]] = value;
		}
	}

	BufferType::IQMatrix(IQMatrix::H264(IQMatrixBufferH264::new(
		scaling_list4x4,
		scaling_list8x8,
	)))
}

/// Ported from cros-codecs `build_pic_param` (BSD-3-Clause).
fn build_pic_param(
	hdr: &SliceHeader,
	pic: &PictureData,
	surface_id: bindings::VASurfaceID,
	dpb: &Dpb<Handle>,
	sps: &Sps,
	pps: &Pps,
) -> BufferType {
	let curr_pic = fill_va_h264_pic(pic, surface_id);

	let short_term = dpb
		.short_term_refs_iter()
		.filter(|entry| !entry.pic.borrow().nonexisting);
	let mut refs: Vec<PictureH264> = short_term
		.chain(dpb.long_term_refs_iter())
		.take(16)
		.map(|entry| fill_va_h264_pic(&entry.pic.borrow(), va_surface_id(&entry.reference)))
		.collect();
	refs.resize_with(16, build_invalid_va_h264_pic);
	let refs: [PictureH264; 16] = refs.try_into().unwrap_or_else(|_| unreachable!("resized to 16"));

	let seq_fields = H264SeqFields::new(
		sps.chroma_format_idc as u32,
		sps.separate_colour_plane_flag as u32,
		sps.gaps_in_frame_num_value_allowed_flag as u32,
		sps.frame_mbs_only_flag as u32,
		sps.mb_adaptive_frame_field_flag as u32,
		sps.direct_8x8_inference_flag as u32,
		// See A.3.3.2.
		(sps.level_idc >= Level::L3_1) as u32,
		sps.log2_max_frame_num_minus4 as u32,
		sps.pic_order_cnt_type as u32,
		sps.log2_max_pic_order_cnt_lsb_minus4 as u32,
		sps.delta_pic_order_always_zero_flag as u32,
	);

	let pic_fields = H264PicFields::new(
		pps.entropy_coding_mode_flag as u32,
		pps.weighted_pred_flag as u32,
		pps.weighted_bipred_idc as u32,
		pps.transform_8x8_mode_flag as u32,
		hdr.field_pic_flag as u32,
		pps.constrained_intra_pred_flag as u32,
		pps.bottom_field_pic_order_in_frame_present_flag as u32,
		pps.deblocking_filter_control_present_flag as u32,
		pps.redundant_pic_cnt_present_flag as u32,
		(pic.nal_ref_idc != 0) as u32,
	);

	let interlaced = !sps.frame_mbs_only_flag as u32;
	let picture_height_in_mbs_minus1 = ((sps.pic_height_in_map_units_minus1 + 1) << interlaced) - 1;

	BufferType::PictureParameter(PictureParameter::H264(PictureParameterBufferH264::new(
		curr_pic,
		refs,
		sps.pic_width_in_mbs_minus1,
		picture_height_in_mbs_minus1,
		sps.bit_depth_luma_minus8,
		sps.bit_depth_chroma_minus8,
		sps.max_num_ref_frames,
		&seq_fields,
		// FMO is not expressible in VA-API.
		0,
		0,
		0,
		pps.pic_init_qp_minus26,
		pps.pic_init_qs_minus26,
		pps.chroma_qp_index_offset,
		pps.second_chroma_qp_index_offset,
		&pic_fields,
		hdr.frame_num,
	)))
}

/// Ported from cros-codecs `fill_ref_pic_list` (BSD-3-Clause).
fn fill_ref_pic_list(list: &[&DpbEntry<Handle>]) -> [PictureH264; 32] {
	let mut pics: Vec<PictureH264> = list
		.iter()
		.take(32)
		.map(|entry| fill_va_h264_pic(&entry.pic.borrow(), va_surface_id(&entry.reference)))
		.collect();
	pics.resize_with(32, build_invalid_va_h264_pic);

	pics.try_into().unwrap_or_else(|_| unreachable!("resized to 32"))
}

/// Ported from cros-codecs `build_slice_param` (BSD-3-Clause).
fn build_slice_param(
	hdr: &SliceHeader,
	slice_size: usize,
	list0: &[&DpbEntry<Handle>],
	list1: &[&DpbEntry<Handle>],
	sps: &Sps,
	pps: &Pps,
) -> BufferType {
	let pwt = &hdr.pred_weight_table;

	let mut luma_weight_l0 = [0i16; 32];
	let mut luma_offset_l0 = [0i16; 32];
	let mut chroma_weight_l0 = [[0i16; 2]; 32];
	let mut chroma_offset_l0 = [[0i16; 2]; 32];

	let mut luma_weight_l1 = [0i16; 32];
	let mut luma_offset_l1 = [0i16; 32];
	let mut chroma_weight_l1 = [[0i16; 2]; 32];
	let mut chroma_offset_l1 = [[0i16; 2]; 32];

	// Explicit weighted prediction: P and SP slices weight list 0 only, B slices
	// weight both. Implicit (weighted_bipred_idc 2) is derived by the hardware.
	let fill_l0 = (pps.weighted_pred_flag && (hdr.slice_type.is_p() || hdr.slice_type.is_sp()))
		|| (pps.weighted_bipred_idc == 1 && hdr.slice_type.is_b());
	let fill_l1 = pps.weighted_bipred_idc == 1 && hdr.slice_type.is_b();
	let chroma = sps.chroma_array_type() != 0;

	if fill_l0 {
		for i in 0..=usize::from(hdr.num_ref_idx_l0_active_minus1) {
			luma_weight_l0[i] = pwt.luma_weight_l0[i];
			luma_offset_l0[i] = i16::from(pwt.luma_offset_l0[i]);
			if chroma {
				for j in 0..2 {
					chroma_weight_l0[i][j] = pwt.chroma_weight_l0[i][j];
					chroma_offset_l0[i][j] = i16::from(pwt.chroma_offset_l0[i][j]);
				}
			}
		}
	}

	if fill_l1 {
		for i in 0..=usize::from(hdr.num_ref_idx_l1_active_minus1) {
			luma_weight_l1[i] = pwt.luma_weight_l1[i];
			luma_offset_l1[i] = i16::from(pwt.luma_offset_l1[i]);
			if chroma {
				for j in 0..2 {
					chroma_weight_l1[i][j] = pwt.chroma_weight_l1[i][j];
					chroma_offset_l1[i][j] = i16::from(pwt.chroma_offset_l1[i][j]);
				}
			}
		}
	}

	BufferType::SliceParameter(SliceParameter::H264(SliceParameterBufferH264::new(
		slice_size as u32,
		0,
		VA_SLICE_DATA_FLAG_ALL,
		hdr.header_bit_size as u16,
		hdr.first_mb_in_slice as u16,
		hdr.slice_type as u8,
		hdr.direct_spatial_mv_pred_flag as u8,
		hdr.num_ref_idx_l0_active_minus1,
		hdr.num_ref_idx_l1_active_minus1,
		hdr.cabac_init_idc,
		hdr.slice_qp_delta,
		hdr.disable_deblocking_filter_idc,
		hdr.slice_alpha_c0_offset_div2,
		hdr.slice_beta_offset_div2,
		fill_ref_pic_list(list0),
		fill_ref_pic_list(list1),
		pwt.luma_log2_weight_denom,
		pwt.chroma_log2_weight_denom,
		fill_l0 as u8,
		luma_weight_l0,
		luma_offset_l0,
		(fill_l0 && chroma) as u8,
		chroma_weight_l0,
		chroma_offset_l0,
		fill_l1 as u8,
		luma_weight_l1,
		luma_offset_l1,
		(fill_l1 && chroma) as u8,
		chroma_weight_l1,
		chroma_offset_l1,
	)))
}

#[cfg(test)]
mod tests {
	use std::os::fd::OwnedFd;

	use super::*;

	/// A gray field with a moving block, so a picture that decoded into the
	/// wrong surface is visible as the block being in the wrong place.
	fn nv12_frame(width: u32, height: u32, step: u32) -> Vec<u8> {
		let (w, h) = (width as usize, height as usize);
		let mut data = vec![128u8; w * h * 3 / 2];
		let x0 = (step * 16) as usize % (w / 2);
		for y in h / 4..h / 2 {
			for x in x0..x0 + w / 4 {
				data[y * w + x] = 235;
			}
		}
		data
	}

	/// Decode to DRM PRIME descriptors and check the pictures never share an
	/// allocation.
	///
	/// The invariant `export` exists to keep: an exported surface is retired
	/// from the pool, because recycling it would have a later picture decoded
	/// over pixels the caller still holds a descriptor for. Two pictures coming
	/// back with the same modifier and layout is expected; two coming back on
	/// the same allocation is the bug.
	///
	/// Skips without a VA-API device, so it is a no-op on a builder and real
	/// coverage on a machine with a GPU. The exported modifier is hardware
	/// policy, so it is printed rather than asserted.
	#[test]
	fn exported_pictures_do_not_share_a_surface() {
		let (width, height) = (320u32, 240u32);
		let Ok(mut encoder) = crate::encode::Encoder::new(crate::encode::Config::new(width, height, 30, 2_000_000, 30))
		else {
			eprintln!("skipping: no VA-API H.264 encoder");
			return;
		};
		let Ok(mut decoder) = Decoder::new(Config::new()) else {
			eprintln!("skipping: no VA-API H.264 decoder");
			return;
		};

		let mut exported = Vec::new();
		for step in 0..8 {
			let unit = encoder
				.encode_nv12(&nv12_frame(width, height, step), step == 0)
				.expect("encode a picture");
			exported.extend(decoder.decode_exported(&unit, step as u64).expect("decode a picture"));
		}
		exported.extend(decoder.flush_exported().expect("flush the decoder"));
		assert!(!exported.is_empty(), "the decoder produced no pictures");

		let first = &exported[0];
		eprintln!(
			"exported {} pictures: {}x{} fourcc {:#x} modifier {:#x}",
			exported.len(),
			first.width,
			first.height,
			first.descriptor.fourcc,
			first.descriptor.objects[0].drm_format_modifier
		);

		let mut seen = Vec::new();
		for (index, frame) in exported.iter().enumerate() {
			assert_eq!((frame.width, frame.height), (width, height));
			assert_eq!(frame.descriptor.fourcc, VA_FOURCC_NV12);
			assert_eq!(frame.descriptor.objects.len(), 1, "a composed export is one object");
			assert_eq!(frame.descriptor.layers[0].num_planes, 2, "NV12 is two planes");

			// Two live descriptors for one allocation would mean the pool handed
			// a retired surface back out. The inode behind a DMA-BUF names the
			// buffer, and is what identifies one allocation from another.
			let inode = inode(&frame.descriptor.objects[0].fd);
			assert!(
				!seen.contains(&inode),
				"picture {index} was decoded into an allocation an earlier one still holds"
			);
			seen.push(inode);
		}
	}

	/// Every picture fed in comes back out, but only once the decoder is flushed.
	///
	/// The contract [`Decoder::flush`] exists for. C.4.5.3 bumps the DPB when a
	/// new picture needs a slot, so a stream that simply stops leaves its tail
	/// sitting there: as many pictures as the sequence's reference and reorder
	/// limits allow, which is one for this crate's IPPP encoder and three for a
	/// stream coded the way x264 does by default. The `decode` half of the
	/// assertion keeps this honest, since a decoder that held nothing back would
	/// satisfy the rest without a flush ever running.
	#[test]
	fn flush_returns_the_pictures_the_dpb_still_holds() {
		const PICTURES: u64 = 5;
		let (width, height) = (320u32, 240u32);

		let Ok(mut encoder) = crate::encode::Encoder::new(crate::encode::Config::new(width, height, 30, 2_000_000, 30))
		else {
			eprintln!("skipping: no VA-API H.264 encoder");
			return;
		};
		let Ok(mut decoder) = Decoder::new(Config::new()) else {
			eprintln!("skipping: no VA-API H.264 decoder");
			return;
		};

		let mut streamed = Vec::new();
		for step in 0..PICTURES {
			let unit = encoder
				.encode_nv12(&nv12_frame(width, height, step as u32), step == 0)
				.expect("encode a picture");
			streamed.extend(decoder.decode(&unit, step).expect("decode a picture"));
		}
		assert!(
			(streamed.len() as u64) < PICTURES,
			"the DPB held nothing back, so this test proves nothing"
		);

		let flushed = decoder.flush().expect("flush the decoder");
		let timestamps: Vec<u64> = streamed
			.iter()
			.chain(&flushed)
			.map(|picture| picture.timestamp)
			.collect();
		assert_eq!(
			timestamps,
			(0..PICTURES).collect::<Vec<_>>(),
			"the stream lost pictures at its end"
		);

		// The bumping process marks what it hands out, so a second flush finds
		// nothing and a caller that flushes twice never sees a picture twice.
		assert!(decoder.flush().expect("flush an empty decoder").is_empty());
	}

	/// The inode of a descriptor, which for a DMA-BUF names the allocation
	/// behind it: every buffer gets its own on the anonymous dma-buf filesystem.
	fn inode(fd: &OwnedFd) -> u64 {
		use std::os::unix::fs::MetadataExt as _;

		// Stat a duplicate, so the descriptor the caller holds is untouched and
		// the `File` wrapper closes only its own copy.
		let file = std::fs::File::from(fd.try_clone().expect("duplicate the descriptor"));
		file.metadata().expect("stat the descriptor").ino()
	}
}
