"""Importing this package registers all built-in tasks."""
from aurelis.tasks.base import Task, get_task, register
from aurelis.tasks.soap import SOAPTask

__all__ = ["Task", "get_task", "register", "SOAPTask"]
