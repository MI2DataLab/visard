import numpy as np
from rectpack import newPacker, PackingBin
from PIL import Image
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt


def pack(sizes_sorted, width):
    """
    Packs squares of given sizes into box of given width (height is calculated).
    Sizes should be sorted.
    """
    rectangles = []
    area = 0
    for val in sizes_sorted:
        rectangles.append((val, val))
        area += val ** 2
    height = np.ceil(area / width)
    bins = [(width, height)]
    packer = newPacker(bin_algo=PackingBin.BBF)

    for r in rectangles:
        packer.add_rect(*r)
    for b in bins:
        packer.add_bin(*b)
    packer.pack()

    return packer, height


def get_size(imp, min_imp, width, max_rect_size):
    """Calculates size of the square on the picture"""
    return round((imp - min_imp) / (1 - min_imp) * width * max_rect_size, 2)


def gen_picture(filenames, importances, save_dir,
                min_imp=0.7, max_rect_size=0.33, width_pixels=3000):
    """
    Generates one picture with all of the pictures given in filenames.
    The size of the picture is calculated by `get_size` function based on
    importance of the picture, min_imp and max_rect_size.
    :param filenames: filenames, where the pictures are stored
    :param importances: impostance of each picture from `filenames`
    :param save_dir: directory where to save the picture
    :param min_imp: minimal importance so the picture is still included
    :param max_rect_size: maximum rectangle size on the picture (between 0 and 1)
    :param width_pixels: the width of the picture in pixels
    """
    # the algorithm doesn't do well with sizes between 0 and 1, it's better to use
    # bigger numbers
    width = 100
    scale = width_pixels / width
    sizes_filenames_dict = dict()
    for i, filename in enumerate(filenames):
        imp = importances[i]
        if imp > min_imp:
            sizes_filenames_dict[filename] = get_size(imp, min_imp, width, max_rect_size)
    sizes_inv = {val: key for key, val in sizes_filenames_dict.items()}
    sizes_filenames_dict = sorted(sizes_filenames_dict.items(),
                                  key=lambda t: (t[1], t[0]))
    sizes_sorted = [x[1] for x in sizes_filenames_dict]
    packer, height = pack(sizes_sorted, width)

    whole_img = np.array(Image.new('RGBA', (int(width * scale), int(height * scale))))
    for i, rect in enumerate(packer[0]):
        size = rect.width
        filename = sizes_inv[size]
        size_scaled = int(scale * size)
        img = Image.open(filename)
        img = np.array(img.resize((size_scaled, size_scaled)))
        x = int(scale * rect.x)
        y = int(scale * rect.y)
        whole_img[y:y + size_scaled, x:x + size_scaled, :] = img
    whole_img = Image.fromarray(whole_img)
    whole_img.save(save_dir)


def calc_MI(x, y, bins):
    """calculates mutual information"""
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def calc_entropy(x, bins):
    hist = np.histogram(x, bins)[0]
    ent = entropy(hist)
    return ent


def calc_importance(data, cols):
    """
    calculates entropy if given one column or
    calculates mutual information if given two columns
    """
    if len(cols) == 1:
        return calc_entropy(data[cols[0]], 20)
    elif len(cols) == 2:
        return calc_MI(data[cols[0]], data[cols[1]], 20)
    else:
        raise ValueError('cols should have one or two column names')


def normalize_importance(importance_dict):
    max_val = max(importance_dict.values())
    min_val = min(importance_dict.values())
    return {key: (val - min_val)/(max_val - min_val) for key, val in importance_dict.items()}


def gen_importance_for_cols(data):
    """
    calculates importance for single columns (based on entropy) and
    pairs of columns (based on mutual information) in given data
    """
    importance_two = {}
    importance_one = {}
    cols = data.columns
    for i, col1 in enumerate(cols):
        for col2 in cols[i:]:
            if col1 != col2:
                importance_two[(col1, col2)] = calc_importance(data, (col1, col2))
    importance_two = normalize_importance(importance_two)
    for col in cols:
        importance_one[(col, )] = calc_importance(data, (col, ))
    importance_one = normalize_importance(importance_one)
    return {**importance_one, **importance_two}


def generate_plot(data, cols, size=7, bins=20):
    """
    generates histogram if given one column or
    generates scatter plot if given two columns
    """
    plt.figure(figsize=(size, size))
    if len(cols) == 1:
        col1 = cols[0]
        plt.hist(data[col1], bins)
        plt.xlabel(col1)
        plt.title(col1 + " Histogram")
        filename = col1 + '.png'
        plt.savefig(filename)
    elif len(cols) == 2:
        col1 = cols[0]
        col2 = cols[1]
        plt.plot(data[col1], data[col2], 'o')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(col1 + " impact on " + col2)
        filename = col1 + '_' + col2 + '.png'
        plt.savefig(filename)
    else:
        raise ValueError('cols should have one or two column names')
    return filename
