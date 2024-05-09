# Import library
from torchvision import transforms as T

# Get transformations based on the input image dimensions
def get_tfs(im_size = (32, 32), imagenet_normalization = True, gray = None):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return T.Compose([T.Resize((im_size)), T.Grayscale(num_output_channels = 3), T.ToTensor(), T.Normalize(mean = mean, std = std)]) if gray else T.Compose([T.Resize((im_size)), T.ToTensor(), T.Normalize(mean = mean, std = std)])