import cv2
from queue import Queue
from threading import Thread

class AsyncVideoWriter:
    def __init__(self, filename, fourcc, fps, frame_size):
        self.queue = Queue()
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.thread = Thread(target=self._write_frames)
        self.thread.daemon = True
        self.thread.start()

    def _write_frames(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                break
            self.writer.write(frame)
            self.queue.task_done()

    def write(self, frame):
        self.queue.put(frame)

    def release(self):
        self.queue.put(None)
        self.thread.join()
        self.writer.release()