import torch


@torch.no_grad()
def get_random_clips(video, clip_size, num_clips=1):
  indices = torch.randint(0, video.shape[1]-clip_size, (num_clips,))
  clips = []
  for idx in indices:
    clips.append(video[:, idx:idx+clip_size])
  return clips


@torch.no_grad()
def get_overlapping_clips(video, clip_size, frames_between_clips, stretch_last_frame=False):
  clip_starts = list(range(0, video.shape[1], frames_between_clips))
  clips = [video[:, s:s+clip_size] for s in clip_starts]

  if stretch_last_frame:
    for i, c in enumerate(clips):
      if c.shape[1] < clip_size:
        missing_frames = clip_size - c.shape[1]
        clips[i] = repeat_last_frame(c, missing_frames)
  return clips


@torch.no_grad()
def repeat_last_frame(clip, n):
  return torch.cat([clip, clip[:, -1:].repeat_interleave(n, dim=1)], dim=1)
