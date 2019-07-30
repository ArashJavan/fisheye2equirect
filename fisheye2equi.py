import argparse
import numpy as np
import cv2 as cv


PATH = "exampleright.jpg"
APERTURE = 210 * np.pi / 180


def lerp(y0, y1, x0, x1, x):
    m = (y1 - y0) / (x1 - x0)
    b = y0
    return m *(x-x0) + b


def fisheye2equi(src_img, size, aperture):

    h_src, w_src = src_img.shape[:2]
    w_dst, h_dst = size

    dst_img = np.zeros((h_dst, w_dst, 3))

    for y in reversed(range(h_dst)):
        y_dst_norm = lerp(-1, 1, 0, h_dst, y)

        for x in range(w_dst):
            x_dst_norm = lerp(-1, 1, 0, w_dst, x)

            longitude = x_dst_norm * np.pi
            latitude = y_dst_norm * np.pi / 2
            p_x = np.cos(latitude) * np.cos(longitude)
            p_y = np.cos(latitude) * np.sin(longitude)
            p_z = np.sin(latitude)

            p_xz = np.sqrt(p_x**2 + p_z**2)
            r = 2 * np.arctan2(p_xz, p_y) / aperture
            theta = np.arctan2(p_z, p_x)
            x_src_norm = r * np.cos(theta)
            y_src_norm = r * np.sin(theta)

            x_src = lerp(0, w_src, -1, 1, x_src_norm)
            y_src = lerp(0, h_src, -1, 1, y_src_norm)

            # supppres out of the bound index error (warning this will overwrite multiply pixels!)
            x_src_ = np.minimum(w_src - 1, np.floor(x_src).astype(np.int))
            y_src_ = np.minimum(h_src - 1, np.floor(y_src).astype(np.int))

            dst_img[y, x, :] = src_img[y_src_, x_src_]
    return dst_img


def run(args):
    src_img_left = cv.imread(args["left"])
    src_img_right = cv.imread(args["right"])

    size = args["dst_size"]
    aperture = args["aperture"] * np.pi / 180

    print("Calculating left image ... ", end="")
    equi_img_left = fisheye2equi(src_img_left, size, aperture)
    print("done!")

    print("Calculating right image ... ", end="")
    equi_img_right = fisheye2equi(src_img_right, size, aperture)
    print("done!")

    print("Merging both images ... ", end="")
    w, h = size
    merged_img = np.zeros((h, w, 3))
    # we should swap the images
    merged_img[:, 0:int(w / 2)] = equi_img_right[:, int(w / 2):]
    merged_img[:, int(w / 2):] = equi_img_left[:, int(w / 2):]
    print("done!")

    cv.imwrite("{}/equirect_right.jpg".format(args["output_dir"]), equi_img_right)
    cv.imwrite("{}/equirect_left.jpg".format(args["output_dir"]), equi_img_left)
    cv.imwrite("{}/equirect.jpg".format(args["output_dir"]), merged_img)
    print("finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from a ROS image-topic.")
    parser.add_argument("-o", "--output-dir", help="Output directory.", action='store', default=".")
    parser.add_argument("-s", "--dst-size", nargs='+', help="size of the output image [width, height]", default=[2000, 1000],  type=int)
    parser.add_argument("-l", "--left", help="left fisheye image")
    parser.add_argument("-r", "--right", help="right fisheye image")
    parser.add_argument("-a", "--aperture", help="aperture of the camera [degree]", default=210, type=float)

    args = vars(parser.parse_args())

    run(args)
    exit(0)
