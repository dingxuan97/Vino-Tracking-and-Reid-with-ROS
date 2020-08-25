class Person:
    def __init__(self):
        self.reid = []
        self.index = []


class Identity:
    def __init__(self, reid_index):
        self.index = reid_index
        self.links = {}
        self.misses = 0

    def update(self, index, face_label):
        if self.index == index:
            if face_label in self.links:
                self.links[face_label] += 1
                self.misses = 0
            elif face_label != 'Unknown':
                self.links[face_label] = 1
            return True
        else:
            self.misses += 1
            return False