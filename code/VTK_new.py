from Script.Vis_2dslice import Slice_2D

class SimpleView():
    def __init__(self):
        pass

    def VisNii(self,label_path,ori_path):
        try:
            Slice_2D.get_volume(self,label_path,ori_path)
            print("Slice_2D.get_volume(self,label_path,ori_path) is done")
        except RuntimeError:
            pass
