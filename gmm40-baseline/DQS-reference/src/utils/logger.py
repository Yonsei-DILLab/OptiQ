import os
import datetime
import re
import numpy as np
from termcolor import colored


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_success", "S", "float"),
	("total_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		# Expand user home and create absolute path to avoid creating literal '~' dirs
		expanded_path = os.path.abspath(os.path.expanduser(dir_path))
		os.makedirs(expanded_path, exist_ok=True)
	except OSError:
		pass
	return expanded_path if 'expanded_path' in locals() else dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, work_dir, wandb, fps=15):
		self._save_dir = make_dir(os.path.join(work_dir, 'eval_video'))
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 0:
			frames = np.stack(self.frames)
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(
		self,
		work_dir,
		seed,
		project,
		entity,
		tags,
		group,
		config,
		disable_wandb=False,
		save_csv=True,
		save_agent=False,
		save_video=False,
		wandb_silent=False,
	):
		self._log_dir = make_dir(work_dir)
		self._model_dir = make_dir(os.path.join(self._log_dir, "models"))
		self._save_csv = save_csv
		self._save_agent = save_agent
		self._group = group
		self._seed = seed
		self._eval = []
		self.project = project
		self.entity = entity
		if disable_wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			save_agent = False
			save_video = False
			self._wandb = None
			self._video = None
			return
		os.environ["WANDB_SILENT"] = "true" if wandb_silent else "false"
		import wandb

		wandb.init(
			project=self.project,
			entity=self.entity,
			name=str(seed),
			group=self._group,
			tags=tags,
			dir=self._log_dir,
			config=config,
		)
		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb
		self._video = (
			VideoRecorder(work_dir, self._wandb)
			if self._wandb and save_video
			else None
		)

	@property
	def video(self):
		return self._video

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = os.path.join(self._model_dir, f'{str(identifier)}.pt')
			agent.save(fp)
			if self._wandb:
				artifact = self._wandb.Artifact(
					self._group + '-' + str(self._seed) + '-' + str(identifier),
					type='model',
				)
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)

	def finish(self, agent=None):
		try:
			self.save_agent(agent)
		except Exception as e:
			print(colored(f"Failed to save model: {e}", "red"))
		if self._wandb:
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.01f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise ValueError(f"invalid log format type: {ty}")

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	def log(self, d, category="train"):
		"""
		Generic log method for scalar metrics (requires 'step').
		Image logging should use log_image.
		"""
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if not self._wandb:
			return
		# Extract step key
		xkey = "step"
		if xkey not in d:
			raise KeyError(f"Missing required key '{xkey}' in log data")
		step = d[xkey]
		# Build wandb log dict
		_wandb_data = {}
		for k, v in d.items():
			if k == xkey:
				continue
			key_name = f"{category}/{k}"
			_wandb_data[key_name] = v
		# Send to wandb
		self._wandb.log(_wandb_data, step=step)

	def log_image(self, name, image, step, category="eval"):
		"""
		Log a single image to wandb under the given category.
		"""
		if not self._wandb:
			return
		key = f"{category}/{name}"
		# Wrap the image in a wandb.Image
		self._wandb.log({key: self._wandb.Image(image)}, step=step)