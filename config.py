import datetime


def get_timestamp():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


class Resolution:
    # OpenCV only allows certain resolutions when recording
    opencv_dimensions = {
        "120p": (160, 120),
        "144p": (176, 144),
        "240p": (320, 240),
        "288p": (352, 288),
        "480p": (640, 480),
        "768p": (1024, 768),
        "1024p": (1280, 1024),
    }

    def __init__(self, *args: str) -> None:
        if len(args) == 1:
            # assume progressive scan value has been passed in
            progressive_scan = args[0]
            self.width = self.opencv_dimensions[progressive_scan][0]
            self.height = self.opencv_dimensions[progressive_scan][1]
        elif len(args) == 2:
            # assume width and height have been passed in
            self.width = args[0]
            self.height = args[1]
        elif len(args) > 2:
            raise ValueError("Instantiating Resolution: Too many arguments")
 
    def get(self):
        return (self.width, self.height)

    def __le__(self, other: 'Resolution'):
        if other is None:
            return False

        if self.width <= other.width and self.height <= other.height:
            # this resolution clearly fits inside the other
            return True
        else:
            # all other cases, this resolution doesn't fully fit inside other
            return False
    
    def __str__(self) -> str:
        return f"({self.width} x {self.height})"