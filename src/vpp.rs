//! Video post-processing: the `VAEntrypointVideoProc` pipeline.
//!
//! The one operation here is an identity blit, which sounds pointless and is
//! not: the destination is a surface the driver allocates for itself, so the
//! blit rewrites the pixels into whatever tiling that driver prefers for a
//! freshly allocated surface. That is how a DMA-BUF carrying a format modifier
//! some other API refuses to import becomes one it accepts.
//!
//! The case this was written for: the Intel iHD H.264 decoder writes
//! `I915_FORMAT_MOD_Y_TILED` (`0x100000000000002`), and Mesa's Vulkan driver
//! does not list that modifier for `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM`, so
//! `vkCreateImage` with an explicit modifier fails and the frame has to go
//! through system memory instead. Blitting it into a VPP-allocated surface on
//! the same device yields `I915_FORMAT_MOD_Y_TILED_GEN12_RC_CCS`
//! (`0x100000000000009`), which Vulkan does list. The pixels never leave the
//! GPU.
//!
//! [`Processor::retile`] is that path end to end: in an exported DMA-BUF, out a
//! surface to export again. [`Processor::import`] and [`Processor::process`] are
//! the two halves, for a caller that already holds a [`Surface`] (a decoder
//! handing over its output) and only needs the second one.
//!
//! ```no_run
//! # fn example(buffer: moq_vaapi::vpp::DmaBuf) -> anyhow::Result<()> {
//! use moq_vaapi::vpp::Processor;
//!
//! let processor = Processor::new("/dev/dri/renderD128")?;
//! let retiled = processor.retile(buffer)?;
//! let exported = retiled.export_prime().map_err(|e| anyhow::anyhow!("export: {e:?}"))?;
//! # let _ = exported;
//! # Ok(())
//! # }
//! ```

use std::os::fd::{AsRawFd, OwnedFd};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use crate::{
	bindings, BufferType, Config as VaConfig, Display, ExternalBufferDescriptor, MemoryType, Picture,
	ProcColorProperties, ProcPipelineParameterBuffer, Surface, SurfaceMemoryDescriptor, UsageHint, VAConfigAttrib,
	VAConfigAttribType, VADRMPRIMESurfaceDescriptor, VAEntrypoint, VAProfile, VA_FOURCC_ARGB, VA_FOURCC_BGRA,
	VA_FOURCC_BGRX, VA_FOURCC_I420, VA_FOURCC_NV12, VA_FOURCC_P010, VA_FOURCC_RGBA, VA_FOURCC_RGBX, VA_FOURCC_XRGB,
	VA_FOURCC_YV12, VA_RT_FORMAT_RGB32, VA_RT_FORMAT_YUV420, VA_RT_FORMAT_YUV420_10,
};

/// One plane of an imported DMA-BUF, within the single object holding them all.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
	/// Byte offset of the plane from the start of the object.
	pub offset: u32,
	/// Bytes between adjacent rows of the plane.
	pub pitch: u32,
}

/// An externally allocated DMA-BUF, described well enough to import as a surface.
///
/// One object holding every plane, which is what a VA-API decoder exports and
/// what a PipeWire producer negotiating a single modifier hands out. A surface
/// split across several objects is not describable here.
///
/// The descriptor owns the file descriptor and keeps it open for the life of the
/// surface built from it. libva does not take ownership (it retrieves the buffer
/// during `vaCreateSurfaces` and leaves the fd alone), so holding it costs one
/// descriptor and removes the question of whether closing it early is safe.
#[derive(Debug)]
pub struct DmaBuf {
	/// Pixel format of the surface, a `VA_FOURCC_*` code.
	pub fourcc: u32,
	/// Pixel format of the layer, a `DRM_FORMAT_*` code. Equal to `fourcc` for
	/// the YUV formats, whose two vocabularies agree, and different for packed
	/// RGB, where DRM names the byte order and VA-API names the channel order.
	pub drm_format: u32,
	/// Format modifier describing the tiling of the object.
	pub modifier: u64,
	/// Width of the allocation in pixels.
	pub width: u32,
	/// Height of the allocation in pixels.
	pub height: u32,
	/// Plane offsets and pitches, in format order. At most four.
	pub planes: Vec<Plane>,
	/// The exported descriptor, held open until the surface is destroyed.
	pub fd: OwnedFd,
}

impl ExternalBufferDescriptor for DmaBuf {
	const MEMORY_TYPE: MemoryType = MemoryType::DrmPrime2;
	type DescriptorAttribute = VADRMPRIMESurfaceDescriptor;

	fn va_surface_attribute(&mut self) -> Self::DescriptorAttribute {
		let mut descriptor = VADRMPRIMESurfaceDescriptor {
			fourcc: self.fourcc,
			width: self.width,
			height: self.height,
			num_objects: 1,
			num_layers: 1,
			..Default::default()
		};

		descriptor.objects[0].fd = self.fd.as_raw_fd();
		// Zero asks the driver to work the size out from the layout, which is
		// what the AMD drivers report on export anyway.
		descriptor.objects[0].size = 0;
		descriptor.objects[0].drm_format_modifier = self.modifier;

		let layer = &mut descriptor.layers[0];
		layer.drm_format = self.drm_format;
		layer.num_planes = self.planes.len().min(4) as u32;
		for (index, plane) in self.planes.iter().take(4).enumerate() {
			layer.object_index[index] = 0;
			layer.offset[index] = plane.offset;
			layer.pitch[index] = plane.pitch;
		}

		descriptor
	}
}

/// The `VA_RT_FORMAT_*` a surface of `fourcc` is allocated as.
///
/// Returns `None` for a format this module has no mapping for, which is the
/// honest answer: guessing the render-target format wrong makes
/// `vaCreateSurfaces` fail with an error that names neither.
pub fn rt_format(fourcc: u32) -> Option<u32> {
	match fourcc {
		VA_FOURCC_NV12 | VA_FOURCC_I420 | VA_FOURCC_YV12 => Some(VA_RT_FORMAT_YUV420),
		VA_FOURCC_P010 => Some(VA_RT_FORMAT_YUV420_10),
		VA_FOURCC_BGRA | VA_FOURCC_BGRX | VA_FOURCC_RGBA | VA_FOURCC_RGBX | VA_FOURCC_ARGB | VA_FOURCC_XRGB => {
			Some(VA_RT_FORMAT_RGB32)
		}
		_ => None,
	}
}

/// A video post-processor on one device.
///
/// Holds the display and the VPP configuration, which are what cost something to
/// build; the per-blit context and surfaces are cheap enough to make and drop
/// around each call, and making them per-call is what keeps one processor usable
/// for frames of different sizes.
pub struct Processor {
	display: Arc<Display>,
	config: VaConfig,
}

impl Processor {
	/// Opens `device` and configures it for video post-processing.
	///
	/// `device` is a DRM render node, e.g. `/dev/dri/renderD128`. Pick the one
	/// matching the device that will consume the result: a re-tile only helps if
	/// both ends are the same GPU.
	pub fn new<P: AsRef<Path>>(device: P) -> anyhow::Result<Self> {
		let device = device.as_ref();
		let display =
			Display::open_drm_display(device).map_err(|e| anyhow::anyhow!("open DRM display {device:?}: {e}"))?;
		Self::with_display(display)
	}

	/// Opens the first device whose driver can post-process, if any.
	///
	/// Mirrors [`Display::open`], and is what a caller with no opinion about
	/// which GPU should use. A machine with more than one wants
	/// [`Processor::new`] instead: a re-tile helps only when the surface ends up
	/// on the device that will read it.
	pub fn open() -> anyhow::Result<Self> {
		for device in crate::DrmDeviceIterator::default() {
			if let Ok(processor) = Self::new(&device) {
				return Ok(processor);
			}
		}
		Err(anyhow::anyhow!("no DRM device supports video post-processing"))
	}

	/// Configures an already open display for video post-processing.
	///
	/// Errors when the device has no `VAEntrypointVideoProc`, which is how a
	/// caller finds out before the first frame rather than during it.
	pub fn with_display(display: Arc<Display>) -> anyhow::Result<Self> {
		// Asking for the attribute is the probe: a device without the VPP
		// entrypoint fails here instead of at vaCreateConfig, with an error that
		// says which entrypoint was missing.
		let mut attrs = [VAConfigAttrib {
			type_: VAConfigAttribType::VAConfigAttribRTFormat,
			value: 0,
		}];
		display
			.get_config_attributes(
				VAProfile::VAProfileNone,
				VAEntrypoint::VAEntrypointVideoProc,
				&mut attrs,
			)
			.map_err(|e| anyhow::anyhow!("query VPP config attributes: {e:?}"))?;

		let config = display
			.create_config(
				attrs.to_vec(),
				VAProfile::VAProfileNone,
				VAEntrypoint::VAEntrypointVideoProc,
			)
			.map_err(|e| anyhow::anyhow!("create VPP config: {e:?}"))?;

		Ok(Self { display, config })
	}

	/// The display this processor runs on, to share with a decoder or an encoder.
	pub fn display(&self) -> &Arc<Display> {
		&self.display
	}

	/// Imports an exported DMA-BUF as a surface, without copying it.
	///
	/// The returned surface owns `buffer`, so the descriptor stays open exactly
	/// as long as the surface referring to it.
	pub fn import(&self, buffer: DmaBuf) -> anyhow::Result<Surface<DmaBuf>> {
		let rt_format =
			rt_format(buffer.fourcc).ok_or_else(|| anyhow::anyhow!("no RT format for fourcc {:#x}", buffer.fourcc))?;
		let (fourcc, width, height) = (buffer.fourcc, buffer.width, buffer.height);

		self.display
			.create_surfaces(rt_format, Some(fourcc), width, height, None, vec![buffer])
			.map_err(|e| anyhow::anyhow!("import DMA-BUF as a surface: {e:?}"))?
			.pop()
			.ok_or_else(|| anyhow::anyhow!("vaCreateSurfaces returned no surface"))
	}

	/// Blits `input` into a driver-allocated surface of `fourcc` and the same size.
	///
	/// The blit is an identity: no scale, no crop, no rotation, and no color
	/// conversion beyond what changing `fourcc` implies. What it does change is
	/// the allocation, since the destination is the driver's own and carries
	/// whatever modifier the driver picks for a surface hinted as
	/// post-processing output that will be exported.
	pub fn process<D: SurfaceMemoryDescriptor>(&self, input: &Surface<D>, fourcc: u32) -> anyhow::Result<Surface<()>> {
		let rt_format = rt_format(fourcc).ok_or_else(|| anyhow::anyhow!("no RT format for fourcc {fourcc:#x}"))?;
		let (width, height) = input.size();

		let mut outputs = self
			.display
			.create_surfaces(
				rt_format,
				Some(fourcc),
				width,
				height,
				Some(UsageHint::USAGE_HINT_VPP_WRITE | UsageHint::USAGE_HINT_EXPORT),
				vec![()],
			)
			.map_err(|e| anyhow::anyhow!("allocate VPP output surface: {e:?}"))?;

		let context = self
			.display
			.create_context(&self.config, width, height, Some(&outputs), true)
			.map_err(|e| anyhow::anyhow!("create VPP context: {e:?}"))?;

		let output = outputs.pop().expect("create_surfaces returned one surface");
		let buffer = context
			.create_buffer(BufferType::ProcPipelineParameter(identity_pipeline(input.id())))
			.map_err(|e| anyhow::anyhow!("create VPP pipeline buffer: {e:?}"))?;

		let mut picture = Picture::new(0, Rc::clone(&context), output);
		picture.add_buffer(buffer);

		let picture = picture
			.begin()
			.map_err(|e| anyhow::anyhow!("vaBeginPicture: {e:?}"))?
			.render()
			.map_err(|e| anyhow::anyhow!("vaRenderPicture: {e:?}"))?
			.end()
			.map_err(|e| anyhow::anyhow!("vaEndPicture: {e:?}"))?
			.sync()
			.map_err(|(e, _)| anyhow::anyhow!("vaSyncSurface: {e:?}"))?;

		picture
			.take_surface()
			.map_err(|_| anyhow::anyhow!("VPP output surface is still referenced"))
	}

	/// Imports `buffer` and blits it into a driver-allocated surface of the same format.
	///
	/// The whole re-tile: export the result with
	/// [`Surface::export_prime`](crate::Surface::export_prime) and read the
	/// modifier off the object to see what the driver chose. Destroying the
	/// returned surface does not invalidate an export taken from it, since the
	/// exported descriptor holds its own reference on the underlying buffer.
	pub fn retile(&self, buffer: DmaBuf) -> anyhow::Result<Surface<()>> {
		let fourcc = buffer.fourcc;
		let input = self.import(buffer)?;
		self.process(&input, fourcc)
	}
}

/// A pipeline parameter buffer describing a plain blit of `surface`.
///
/// Every knob at its neutral value: no regions (so the whole surface), no
/// filters, no references, no rotation or mirroring, and default color
/// properties, which leave the color standard as the driver reads it off the
/// surface rather than asking for a conversion.
fn identity_pipeline(surface: bindings::VASurfaceID) -> ProcPipelineParameterBuffer {
	ProcPipelineParameterBuffer::new(
		surface,
		None,
		0,
		None,
		0,
		0,
		0,
		0,
		None,
		None,
		None,
		0,
		None,
		0,
		None,
		0,
		0,
		ProcColorProperties::default(),
		ProcColorProperties::default(),
		0,
		None,
	)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Round-trip a driver-allocated NV12 surface through the post-processor and
	/// report what the modifier did.
	///
	/// Skips without a VA-API device, so it is a no-op on a builder and real
	/// coverage on a machine with a GPU. What it asserts is that the whole
	/// sequence runs (import an exported DMA-BUF, blit, export the result); the
	/// modifier the driver picks is hardware policy, so it is printed rather
	/// than asserted.
	#[test]
	fn retile_round_trip() {
		let Some(display) = Display::open() else {
			eprintln!("skipping: no VA-API display");
			return;
		};
		let Ok(processor) = Processor::with_display(display) else {
			eprintln!("skipping: device has no video post-processing entrypoint");
			return;
		};

		// A decode-hinted surface, which is the allocation whose tiling is the
		// problem this module exists for.
		let mut surfaces = processor
			.display()
			.create_surfaces(
				VA_RT_FORMAT_YUV420,
				Some(VA_FOURCC_NV12),
				640,
				480,
				Some(UsageHint::USAGE_HINT_DECODER | UsageHint::USAGE_HINT_EXPORT),
				vec![()],
			)
			.expect("allocate a decode surface");
		let decoded = surfaces.pop().expect("one surface");
		let exported = decoded.export_prime().expect("export the decode surface");

		assert_eq!(exported.objects.len(), 1, "a composed export is one object");
		let object = &exported.objects[0];
		let layer = &exported.layers[0];
		let source_modifier = object.drm_format_modifier;
		let buffer = DmaBuf {
			fourcc: exported.fourcc,
			drm_format: layer.drm_format,
			modifier: source_modifier,
			width: exported.width,
			height: exported.height,
			planes: (0..layer.num_planes as usize)
				.map(|i| Plane {
					offset: layer.offset[i],
					pitch: layer.pitch[i],
				})
				.collect(),
			// The export is composed into one object, so the first fd is the
			// whole surface. Any others belong to a layout this type cannot
			// describe, and leaving them to `exported` closes them on drop.
			fd: exported.objects[0]
				.fd
				.try_clone()
				.expect("clone the exported descriptor"),
		};

		let retiled = processor.retile(buffer).expect("re-tile the decode surface");
		let output = retiled.export_prime().expect("export the re-tiled surface");
		let target_modifier = output.objects[0].drm_format_modifier;

		eprintln!("modifier {source_modifier:#x} -> {target_modifier:#x}");
		assert_eq!(output.fourcc, exported.fourcc, "the blit preserves the pixel format");
		assert_eq!((output.width, output.height), (exported.width, exported.height));
		assert_eq!(output.objects.len(), 1, "a composed export is one object");
		assert_eq!(
			output.layers[0].num_planes, layer.num_planes,
			"the blit preserves the plane count"
		);
	}
}
