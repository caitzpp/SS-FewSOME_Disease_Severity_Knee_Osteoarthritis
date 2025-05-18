from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom ImageFolder that includes file paths."""
    def __getitem__(self, index):
        # Get the original tuple (image, label)
        original_tuple = super().__getitem__(index)
        
        # Get the image file path
        path = self.imgs[index][0]
        
        # Return the original tuple with the path
        return original_tuple + (path,)