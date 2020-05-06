import numpy as np
import scipy as sp
import scipy.signal  # one option for a 2D convolution library
import cv2
from numba import jit
import sys


class SeamCarving:
    # input_filename, output_filename, dimension_change, energy_algorithm
    def __init__(self, input_filename, output_filename, dimension_change, energy_algorithm, display_seams):
        self.input_filename = input_filename
        input_img = cv2.imread(input_filename, cv2.IMREAD_COLOR)
        dimension_change = dimension_change.split(', ')
        new_dimensions = (input_img.shape[0] + int(dimension_change[0]), input_img.shape[1] + int(dimension_change[1]))
        self.energy_algorithm = energy_algorithm
        self.output_filename = output_filename
        self.seam_display = display_seams

        self.input_img = np.atleast_3d(input_img).astype(np.float)
        self.target_dimensions = new_dimensions
        self.new_image = self.input_img
        self.copy_image = self.input_img
        self.energy_map = np.ndarray(new_dimensions, dtype=float)
        self.cme_map = np.ndarray(new_dimensions, dtype=float)
        self.seam = np.ndarray(new_dimensions, dtype=float)
        self.seam_collection = np.ndarray(new_dimensions, dtype=float)
        self.seam_count = 0
        self.removal_seam_collection = {}

        print('Seam Carving Started!')
        self.aspect_ratio_change()

    def generate_energy_fnx(self):
        b_channel, g_channel, r_channel = cv2.split(self.new_image)

        # sobel edge
        b_energy = np.absolute(cv2.Sobel(b_channel, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(b_channel, cv2.CV_64F, 0, 1, ksize=3))
        g_energy = np.absolute(cv2.Sobel(g_channel, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(g_channel, cv2.CV_64F, 0, 1, ksize=3))
        r_energy = np.absolute(cv2.Sobel(r_channel, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(r_channel, cv2.CV_64F, 0, 1, ksize=3))

        # np.gradient
        # b_energy = np.gradient(b_channel, axis=0) + np.gradient(b_channel, axis=1)
        # g_energy = np.gradient(g_channel, axis=0) + np.gradient(g_channel, axis=1)
        # r_energy = np.gradient(r_channel, axis=0) + np.gradient(r_channel, axis=1)

        total_energy = b_energy + g_energy + r_energy

        # debug
        cv2.imwrite('energy.png', np.dstack((b_energy, g_energy, r_energy)))
        return total_energy

    def cumulative_minimum_energy_backward(self):
        image_height, image_width, _ = self.new_image.shape
        self.cme_map = np.zeros((image_height, image_width), dtype=np.dtype('float'))
        self.cme_map[0] = self.energy_map[0]
        for row in np.arange(1, image_height):
            for column in np.arange(0, image_width):
                if column == 0:
                    # look at top, right 2 neighbors
                    self.cme_map[row, column] = self.energy_map[row, column] + min(
                        self.cme_map[row - 1, column], self.cme_map[row - 1, column + 1])
                elif column == image_width - 1:
                    # look at top, left 2 neighbors
                    self.cme_map[row, column] = self.energy_map[row, column] + min(
                        self.cme_map[row - 1, column - 1], self.cme_map[row - 1, column])
                else:
                    # look at top 3 neighbors
                    self.cme_map[row, column] = self.energy_map[row, column] + min(
                        self.cme_map[row - 1, column - 1], self.cme_map[row - 1, column], self.cme_map[row - 1, column + 1])
        return self.cme_map

    def cumulative_minimum_energy_forward(self):
        image_height, image_width, channels = self.new_image.shape
        self.cme_map = np.zeros((image_height, image_width), dtype=np.dtype('float'))
        self.cme_map[0] = self.energy_map[0]
        for channel in np.arange(0, channels):
            for row in np.arange(1, image_height):
                for column in np.arange(0, image_width):
                    C_L, C_U, C_R = (0, 0, 0)
                    if column == 0:
                        # look at top, right 2 neighbors
                        C_L = np.absolute(
                            self.new_image[row, column + 1, channel] - 0) + \
                              np.absolute(self.new_image[row - 1, column, channel] - 0)
                        C_U = np.absolute(self.new_image[row, column + 1, channel] - 0)
                        C_R = np.absolute(self.new_image[row, column + 1, channel] - 0) + \
                              np.absolute(
                                  self.new_image[row - 1, column, channel] - self.new_image[row, column + 1, channel])
                        self.cme_map[row, column] = min(self.cme_map[row - 1, column] + C_U,
                                                        self.cme_map[row - 1, column + 1] + C_R)
                    elif column == image_width - 1:
                        # look at top, left 2 neighbors
                        C_L = np.absolute(0 - self.new_image[row, column - 1, channel]) + \
                              np.absolute(
                                  self.new_image[row - 1, column, channel] - self.new_image[row, column - 1, channel])
                        C_U = np.absolute(0 - self.new_image[row, column - 1, channel])
                        C_R = np.absolute(0 - self.new_image[row, column - 1, channel]) + \
                              np.absolute(self.new_image[row - 1, column, channel] - 0)
                        self.cme_map[row, column] = min(self.cme_map[row - 1, column - 1] + C_L,
                                                        self.cme_map[row - 1, column] + C_U)
                    else:
                        # look at top 3 neighbors
                        C_L = np.absolute(
                            self.new_image[row, column + 1, channel] - self.new_image[row, column - 1, channel]) + \
                              np.absolute(
                                  self.new_image[row - 1, column, channel] - self.new_image[row, column - 1, channel])
                        C_U = np.absolute(
                            self.new_image[row, column + 1, channel] - self.new_image[row, column - 1, channel])
                        C_R = np.absolute(
                            self.new_image[row, column + 1, channel] - self.new_image[row, column - 1, channel]) + \
                              np.absolute(
                                  self.new_image[row - 1, column, channel] - self.new_image[row, column + 1, channel])

                        self.cme_map[row, column] = min(self.cme_map[row - 1, column - 1] + C_L,
                                                        self.cme_map[row - 1, column] + C_U,
                                                        self.cme_map[row - 1, column + 1] + C_R)
        return self.cme_map

    def find_seam(self):
        seam_cost = min(self.cme_map[-1])
        map_height, map_width = self.cme_map.shape
        vertical_seam_path = np.zeros((map_height, map_width), dtype=np.dtype('float'))

        # find seam start point from last row in energy map
        seam_start_point = list(self.cme_map[-1]).index(min(self.cme_map[-1]))
        vertical_seam_path[map_height - 1, seam_start_point] = 12345

        # finish seam through 8-connected path of pixels, bottom up
        for row in np.arange(map_height - 1, 0, -1):
            if seam_start_point == map_width - 1:
                neighbor_pixels = {
                    self.cme_map[row - 1, seam_start_point - 1]: seam_start_point - 1,
                    self.cme_map[row - 1, seam_start_point]: seam_start_point
                }
            elif seam_start_point == 0:
                neighbor_pixels = {
                    self.cme_map[row - 1, seam_start_point]: seam_start_point,
                    self.cme_map[row - 1, seam_start_point + 1]: seam_start_point + 1
                }
            else:
                neighbor_pixels = {
                    self.cme_map[row - 1, seam_start_point - 1]: seam_start_point - 1,
                    self.cme_map[row - 1, seam_start_point]: seam_start_point,
                    self.cme_map[row - 1, seam_start_point + 1]: seam_start_point + 1
                }
            neighbor_temp = min(neighbor_pixels)
            seam_next_point = neighbor_pixels[neighbor_temp]
            seam_cost += neighbor_temp
            vertical_seam_path[row - 1, seam_next_point] = 12345
            seam_start_point = seam_next_point
        # cv2.imwrite('vertical_path.png', vertical_seam_path)
        return vertical_seam_path, seam_cost

    def remove_seam(self):
        seam_height, seam_width = self.seam.shape
        b_channel, g_channel, r_channel = cv2.split(self.new_image)
        new_slice_b = np.delete(b_channel[0], np.where(self.seam[0] == 12345))
        new_slice_g = np.delete(g_channel[0], np.where(self.seam[0] == 12345))
        new_slice_r = np.delete(r_channel[0], np.where(self.seam[0] == 12345))

        for row in np.arange(1, seam_height):
            row_slice_b = np.delete(b_channel[row], np.where(self.seam[row] == 12345))
            row_slice_g = np.delete(g_channel[row], np.where(self.seam[row] == 12345))
            row_slice_r = np.delete(r_channel[row], np.where(self.seam[row] == 12345))
            new_slice_b = np.vstack((new_slice_b, row_slice_b))
            new_slice_g = np.vstack((new_slice_g, row_slice_g))
            new_slice_r = np.vstack((new_slice_r, row_slice_r))

            # draws seams
            self.copy_image[row, np.where(self.seam[row] == 12345)] = (0, 0, 255)

        self.seam_count += 1
        return np.dstack((new_slice_b, new_slice_g, new_slice_r))

    def store_removal_seam(self):
        seam_collection = np.ndarray(self.target_dimensions, dtype=float)
        seam_height, seam_width = self.seam.shape
        first_seam_pixel = np.where(self.seam[0] == 12345)[0]
        seam_collection[0][first_seam_pixel] = 12345
        b_channel, g_channel, r_channel = cv2.split(self.new_image)
        new_slice_b = np.delete(b_channel[0], first_seam_pixel)
        new_slice_g = np.delete(g_channel[0], first_seam_pixel)
        new_slice_r = np.delete(r_channel[0], first_seam_pixel)

        for row in np.arange(1, seam_height):
            seam_location = np.where(self.seam[row] == 12345)
            seam_collection[row][seam_location] = 12345

            row_slice_b = np.delete(b_channel[row], seam_location)
            row_slice_g = np.delete(g_channel[row], seam_location)
            row_slice_r = np.delete(r_channel[row], seam_location)
            new_slice_b = np.vstack((new_slice_b, row_slice_b))
            new_slice_g = np.vstack((new_slice_g, row_slice_g))
            new_slice_r = np.vstack((new_slice_r, row_slice_r))

        self.seam_count += 1

        return np.dstack((new_slice_b, new_slice_g, new_slice_r)), seam_collection

    def add_seam_from_collection(self, seam):
        b_channel, g_channel, r_channel = cv2.split(self.copy_image)
        seam_location = np.where(seam[0] == 12345)[0]
        b_aggregate = np.insert(b_channel[0], seam_location, b_channel[0][seam_location])
        g_aggregate = np.insert(g_channel[0], seam_location, g_channel[0][seam_location])
        r_aggregate = np.insert(r_channel[0], seam_location, r_channel[0][seam_location])

        for row in np.arange(0, seam.shape[0]):
            if row == 0:
                continue
            seam_location = np.where(seam[row] == 12345)[0]
            b_slice = np.insert(b_channel[row], seam_location, b_channel[row][seam_location])
            g_slice = np.insert(g_channel[row], seam_location, g_channel[row][seam_location])
            r_slice = np.insert(r_channel[row], seam_location, r_channel[row][seam_location])

            b_aggregate = np.vstack((b_aggregate, b_slice))
            g_aggregate = np.vstack((g_aggregate, g_slice))
            r_aggregate = np.vstack((r_aggregate, r_slice))

        return np.dstack((b_aggregate, g_aggregate, r_aggregate))

    def add_seam(self):
        seam_height, seam_width = self.seam.shape
        b_channel, g_channel, r_channel = cv2.split(self.new_image)
        first_seam_pixel = np.where(self.seam[0] == 12345)[0]
        self.seam_collection[0][first_seam_pixel] = 12345
        if first_seam_pixel == seam_width - 1:
            neighbor_avg_b = (b_channel[0][first_seam_pixel - 1] + b_channel[0][first_seam_pixel]) / 2.0
            neighbor_avg_g = (g_channel[0][first_seam_pixel - 1] + g_channel[0][first_seam_pixel]) / 2.0
            neighbor_avg_r = (r_channel[0][first_seam_pixel - 1] + r_channel[0][first_seam_pixel]) / 2.0
        elif first_seam_pixel == 0:
            neighbor_avg_b = (b_channel[0][first_seam_pixel] + b_channel[0][first_seam_pixel + 1]) / 2.0
            neighbor_avg_g = (g_channel[0][first_seam_pixel] + g_channel[0][first_seam_pixel + 1]) / 2.0
            neighbor_avg_r = (r_channel[0][first_seam_pixel] + r_channel[0][first_seam_pixel + 1]) / 2.0
        else:
            neighbor_avg_b = (b_channel[0][first_seam_pixel - 1] + b_channel[0][first_seam_pixel + 1]) / 2.0
            neighbor_avg_g = (g_channel[0][first_seam_pixel - 1] + g_channel[0][first_seam_pixel + 1]) / 2.0
            neighbor_avg_r = (r_channel[0][first_seam_pixel - 1] + r_channel[0][first_seam_pixel + 1]) / 2.0

        neighbor_avg_b = np.insert(b_channel[0], first_seam_pixel + 1, neighbor_avg_b)
        neighbor_avg_g = np.insert(g_channel[0], first_seam_pixel + 1, neighbor_avg_g)
        neighbor_avg_r = np.insert(r_channel[0], first_seam_pixel + 1, neighbor_avg_r)

        for row in np.arange(1, seam_height):
            seam_location = np.where(self.seam[row] == 12345)[0]
            self.seam_collection[row][seam_location] = 12345
            if seam_location == seam_width - 1:
                neighbor_avg_b_slice = (b_channel[row][seam_location - 1] + b_channel[row][seam_location]) / 2.0
                neighbor_avg_g_slice = (g_channel[row][seam_location - 1] + g_channel[row][seam_location]) / 2.0
                neighbor_avg_r_slice = (r_channel[row][seam_location - 1] + r_channel[row][seam_location]) / 2.0
            elif seam_location == 0:
                neighbor_avg_b_slice = (b_channel[row][seam_location] + b_channel[row][seam_location + 1]) / 2.0
                neighbor_avg_g_slice = (g_channel[row][seam_location] + g_channel[row][seam_location + 1]) / 2.0
                neighbor_avg_r_slice = (r_channel[row][seam_location] + r_channel[row][seam_location + 1]) / 2.0
            else:
                neighbor_avg_b_slice = (b_channel[row][seam_location - 1] + b_channel[row][seam_location + 1]) / 2.0
                neighbor_avg_g_slice = (g_channel[row][seam_location - 1] + g_channel[row][seam_location + 1]) / 2.0
                neighbor_avg_r_slice = (r_channel[row][seam_location - 1] + r_channel[row][seam_location + 1]) / 2.0

            neighbor_avg_b_slice = np.insert(b_channel[row], seam_location + 1, neighbor_avg_b_slice)
            neighbor_avg_g_slice = np.insert(g_channel[row], seam_location + 1, neighbor_avg_g_slice)
            neighbor_avg_r_slice = np.insert(r_channel[row], seam_location + 1, neighbor_avg_r_slice)

            neighbor_avg_b = np.vstack((neighbor_avg_b, neighbor_avg_b_slice))
            neighbor_avg_g = np.vstack((neighbor_avg_g, neighbor_avg_g_slice))
            neighbor_avg_r = np.vstack((neighbor_avg_r, neighbor_avg_r_slice))

        self.seam_count += 1
        return np.dstack((neighbor_avg_b, neighbor_avg_g, neighbor_avg_r))

    def aspect_ratio_change(self):
        target_height, target_width = self.target_dimensions
        delta_height = target_height - self.input_img.shape[0]
        delta_width = target_width - self.input_img.shape[1]

        if delta_height < 0 and delta_width < 0:            # reduce height and width
            self.seam_removal_decision('both')
        elif delta_height < 0 and delta_width > 0:          # reduce height, expand width
            pass
        elif delta_height > 0 and delta_width < 0:          # expand height, reduce width
            pass
        elif delta_height > 0 and delta_width > 0:          # expand height, expand width
            pass
        elif delta_height == 0 and delta_width < 0:
            self.seam_removal_decision('vertical')
        elif delta_height == 0 and delta_width > 0:
            self.seam_addition_decision('vertical')
        elif delta_height < 0 and delta_width == 0:
            self.seam_removal_decision('horizontal')
        elif delta_height > 0 and delta_width == 0:
            self.seam_addition_decision('horizontal')

    def generate_transport_map(self, horizontal_seam_cost, vertical_seam_cost):

        self.new_image = np.rot90(self.new_image, -1)
        self.energy_map = self.generate_energy_fnx()
        self.cme_map = self.cumulative_minimum_energy_backward()
        self.seam, horizontal_seam_cost = self.find_seam()
        self.new_image = np.rot90(self.new_image, 1)

        self.energy_map = self.generate_energy_fnx()
        self.cme_map = self.cumulative_minimum_energy_backward()
        self.seam, vertical_seam_cost = self.find_seam()

        # stopped here
        r = self.input_img.shape[1] - self.target_dimensions[1]
        c = self.input_img.shape[0] - self.target_dimensions[0]

        transport_map = np.ndarray(shape=(r, c), dtype=float)
        transport_map[0, 0] = 0
        bit_map = np.ndarray(shape=(r, c), dtype=int)

        for row in np.arange(0, r):
            for column in np.arange(0, c):
                if row == 0 and column == 0:
                    continue

                top_neighbor = transport_map[row - 1, column] + vertical_seam_cost
                left_neighbor = transport_map[row, column - 1] + horizontal_seam_cost
                transport_map[row, column] = min(top_neighbor, left_neighbor)
                bit_map[row, column] = np.argmin((top_neighbor, left_neighbor))

        return transport_map

    def seam_removal_decision(self, which_seam):
        if which_seam is 'vertical':             # vertical seam
            while self.new_image.shape[1] != self.target_dimensions[1]:
                self.energy_map = self.generate_energy_fnx()
                if self.energy_algorithm == 'backward':
                    self.cme_map = self.cumulative_minimum_energy_backward()
                elif self.energy_algorithm == 'forward':
                    self.cme_map = self.cumulative_minimum_energy_forward()
                else:
                    print('energy algorithm error: {}, {}'.format(self.energy_algorithm, type(self.energy_algorithm)))
                    return
                self.seam, _ = self.find_seam()
                self.new_image = self.remove_seam()
                print('Processed seam {}'.format(self.seam_count))
        elif which_seam is 'horizontal':
            while self.new_image.shape[0] != self.target_dimensions[0]:
                self.new_image = np.rot90(self.new_image, -1)
                self.energy_map = self.generate_energy_fnx()
                if self.energy_algorithm == 'backward':
                    self.cme_map = self.cumulative_minimum_energy_backward()
                elif self.energy_algorithm == 'forward':
                    self.cme_map = self.cumulative_minimum_energy_forward()
                else:
                    print('energy algorithm error: {}, {}'.format(self.energy_algorithm, type(self.energy_algorithm)))
                    return
                self.seam, _ = self.find_seam()
                self.new_image = self.remove_seam()
                self.new_image = np.rot90(self.new_image, 1)
                print('Processed seam {}'.format(self.seam_count))
        elif which_seam == 'both':
            index = 0
            while self.new_image.shape[0] != self.target_dimensions[0] and \
                    self.new_image.shape[1] != self.target_dimensions[1]:
                #transport_map = self.generate_transport_map()
                self.new_image = self.remove_seam()
                self.new_image = np.rot90(self.new_image, 1)
                print('Processed seam {}'.format(index))
                index += 1
        cv2.imwrite(self.output_filename, self.new_image)

        if self.seam_display:
            seam_filename = self.output_filename.split('.png')[0] + '-seams.png'
            cv2.imwrite(seam_filename, self.copy_image)
            cv2.imshow('Seams: {}'.format(seam_filename), cv2.imread(seam_filename))

        cv2.imshow('Original: {}'.format(self.input_filename), cv2.imread(self.input_filename))
        cv2.imshow('Processed: {}'.format(self.output_filename), cv2.imread(self.output_filename))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def seam_addition_decision(self, which_seam):
        if which_seam is 'vertical':
            for _ in range(self.target_dimensions[1] - self.new_image.shape[1]):
                self.energy_map = self.generate_energy_fnx()
                if self.energy_algorithm == 'backward':
                    self.cme_map = self.cumulative_minimum_energy_backward()
                elif self.energy_algorithm == 'forward':
                    self.cme_map = self.cumulative_minimum_energy_forward()
                else:
                    print('energy algorithm error: {}, {}'.format(self.energy_algorithm, type(self.energy_algorithm)))
                    return
                self.seam, _ = self.find_seam()

                # average seam addition
                # self.new_image = self.add_seam()
                # print('Processed seam {}'.format(self.seam_count))

                # back in time seam addition
                self.new_image, self.removal_seam_collection[self.seam_count] = self.store_removal_seam()
                print('Storing seam {} back in time'.format(self.seam_count))

            for key, seam in self.removal_seam_collection.items():
                self.new_image = self.add_seam_from_collection(seam)
                print('Applying seam {}'.format(key))
                self.copy_image = self.new_image

        elif which_seam is 'horizontal':
            index = 0
            while self.new_image.shape[0] != self.target_dimensions[0]:
                self.new_image = np.rot90(self.new_image, -1)
                self.energy_map = self.generate_energy_fnx()
                if self.energy_algorithm == 'backward':
                    self.cme_map = self.cumulative_minimum_energy_backward()
                elif self.energy_algorithm == 'forward':
                    self.cme_map = self.cumulative_minimum_energy_forward()
                else:
                    print('energy algorithm error: {}, {}'.format(self.energy_algorithm, type(self.energy_algorithm)))
                    return
                self.seam, _ = self.find_seam()
                self.new_image = self.add_seam()
                self.new_image = np.rot90(self.new_image, 1)
                print('Processed seam {}'.format(index))
                index += 1
        elif which_seam == 'both':
            index = 0
            while self.new_image.shape[0] != self.target_dimensions[0] and \
                    self.new_image.shape[1] != self.target_dimensions[1]:
                #transport_map = self.generate_transport_map()
                self.new_image = self.remove_seam()
                self.new_image = np.rot90(self.new_image, 1)
                print('Processed seam {}'.format(index))
                index += 1
        cv2.imwrite(self.output_filename, self.new_image)

        if self.seam_display:
            self.display_seams()
            seam_filename = self.output_filename.split('.png')[0] + '-seams.png'
            cv2.imwrite(seam_filename, self.new_image)
            cv2.imshow('Seams: {}'.format(seam_filename), cv2.imread(seam_filename))

        cv2.imshow('Original: {}'.format(self.input_filename), cv2.imread(self.input_filename))
        cv2.imshow('Processed: {}'.format(self.output_filename), cv2.imread(self.output_filename))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_seams(self):
        # collection_height, collection_width = self.seam_collection.shape
        # for row in np.arange(0, collection_height):
        #     for column in np.arange(0, collection_width):
        #         if self.seam_collection[row][column] == 12345:
        #             self.new_image[row][column] = (0, 0, 255)

        seam_height, seam_width = self.removal_seam_collection[1].shape
        for key, seam in self.removal_seam_collection.items():
            for row in np.arange(0, seam_height):
                for column in np.arange(0, seam_width):
                    if seam[row][column] == 12345:
                        self.new_image[row][column] = (0, 0, 255)


def main():
    # command line arguments
    # 1. Input Filename
    # 2. Output Filename
    # 3. Dimension Change: +50, -50
    # 4. Forward or Backward Energy

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    dimension_change = sys.argv[3]
    energy_algorithm = sys.argv[4]
    display_seams = sys.argv[5]

    SeamCarving(input_filename, output_filename, dimension_change, energy_algorithm, display_seams)


if __name__ == "__main__":
    main()
