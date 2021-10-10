import threading
import matplotlib.pyplot as plt
import numpy as np

class Drawer:
    def __init__(self, title='Plot'):
        self.update_plot = threading.Event()
        self.update_plot.set()
        self.stopped = False
        self.values = []
        self.title = title

    def create_thread(self):
        self.thread = threading.Thread(target=self.plot)
        self.thread.daemon = True
        self.thread.start()

    def interactive_start(self, onpress):
        def press(event):
            print(':', event.key)
            onpress(event.key)
            ax.clear()
            ax.plot(np.arange(len(self.values)), self.values)
            fig.canvas.draw()
        fig, ax = plt.subplots()
        ax.title.set_text(self.title)
        ax.plot(np.arange(len(self.values)), self.values)
        ax.set_xlabel('Episode')
        fig.canvas.mpl_connect('key_press_event', press)
        plt.show()

    def add_value(self, v):
        self.values.append(v)
        self.update_plot.set()

    def stop(self):
        self.stopped = True
        self.update_plot.set()
        plt.ioff()

    def render(self, ax=plt):
        plt.cla()
        plt.title(self.title)
        ax.plot(np.arange(len(self.values)), self.values)
        # plt.ylim(-2000, 0)
        ax.xlabel('Episode')
        plt.draw()

    def plot(self):
        plt.ion()
        while not self.stopped:
            self.update_plot.wait()
            self.update_plot.clear()
            self.render()
            plt.pause(0.2)
        plt.ioff()
        plt.close()

