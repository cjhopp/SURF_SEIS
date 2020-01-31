import pyinotify

class EventHandler(pyinotify.ProcessEvent):
    def process_IN_CLOSE_NOWRITE(self, event):
        print("File was closed without writing: " + event.pathname)
    def process_IN_CLOSE_WRITE(self, event):
        print("File was closed with writing: " + event.pathname)

def watch(filename):
    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CLOSE_NOWRITE | pyinotify.IN_CLOSE_WRITE
    wm.add_watch(filename, mask)

    eh = EventHandler()
    notifier = pyinotify.Notifier(wm, eh)
    notifier.loop()

if __name__ == '__main__':
    watch('/home/sigmav/test')
