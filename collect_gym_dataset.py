# From https://github.com/google-research/planet/blob/0c6f7d3c56fe691da5b0a2fc62db3cb7075cfcf4/planet/control/wrappers.py#L427

import datetime
import io
import os
import uuid

import numpy as np
import tensorflow as tf

class CollectGymDataset(object):
  """Collect transition tuples and store episodes as Numpy files."""

  def __init__(self, env, outdir):
    self._env = env
    self._outdir = outdir and os.path.expanduser(outdir)
    self._episode = None
    self._transition = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action, *args, **kwargs):
    if kwargs.get('blocking', True):
      transition = self._env.step(action, *args, **kwargs)
      return self._process_step(action, *transition)
    else:
      future = self._env.step(action, *args, **kwargs)
      return lambda: self._process_step(action, *future())

  def reset(self, *args, **kwargs):
    if kwargs.get('blocking', True):
      observ = self._env.reset(*args, **kwargs)
      return self._process_reset(observ)
    else:
      future = self._env.reset(*args, **kwargs)
      return lambda: self._process_reset(future())

  def _process_step(self, action, observ, reward, done, info):
    self._transition.update({'action': action, 'reward': reward})
    self._transition.update(info)
    self._episode.append(self._transition)
    self._transition = {}
    if not done:
      self._transition.update(self._process_observ(observ))
    else:
      episode = self._get_episode()
      info['episode'] = episode
      if self._outdir:
        filename = self._get_filename()
        self._write(episode, filename)
    return observ, reward, done, info

  def _process_reset(self, observ):
    self._episode = []
    self._transition = {}
    self._transition.update(self._process_observ(observ))
    return observ

  def _process_observ(self, observ):
    if not isinstance(observ, dict):
      observ = {'observ': observ}
    return observ

  def _get_filename(self):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4()).replace('-', '')
    filename = '{}-{}.npz'.format(timestamp, identifier)
    filename = os.path.join(self._outdir, filename)
    return filename

  def _get_episode(self):
    episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
    episode = {k: np.array(v) for k, v in episode.items()}
    for key, sequence in episode.items():
      if sequence.dtype == 'object':
        message = "Sequence '{}' is not numeric:\n{}"
        raise RuntimeError(message.format(key, sequence))
    return episode

  def _write(self, episode, filename):
    if not tf.gfile.Exists(self._outdir):
      tf.gfile.MakeDirs(self._outdir)
    with io.BytesIO() as file_:
      np.savez_compressed(file_, **episode)
      file_.seek(0)
      with tf.gfile.Open(filename, 'w') as ff:
        ff.write(file_.read())
    name = os.path.splitext(os.path.basename(filename))[0]
    print('Recorded episode {}.'.format(name))
