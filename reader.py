# Most of the code refactored from https://github.com/ewino/qreader; license as specified in that repo.
__author__ = "ewino"

from collections.abc import Iterator
from itertools import permutations


def is_rect_overlapping(rect1, rect2):
    h_overlaps = is_range_overlapping((rect1[0], rect1[2]), (rect2[0], rect2[2]))
    v_overlaps = is_range_overlapping((rect1[1], rect1[3]), (rect2[1], rect2[3]))
    return h_overlaps and v_overlaps


def is_range_overlapping(a, b):
    """Neither range is completely greater than the other
    :param tuple a: first range
    :param tuple b: second range
    """
    return (a[0] <= b[1]) and (b[0] <= a[1])


def ints_to_bytes(ints):
    return bytes(ints)


# QR error correct levels
ERROR_CORRECT_L = 0
ERROR_CORRECT_M = 1
ERROR_CORRECT_Q = 2
ERROR_CORRECT_H = 3

# QR encoding modes (based on qrcode package)
MODE_NUMBER = 1
MODE_ALPHA_NUM = 2
MODE_BYTES = 4
MODE_KANJI = 8
MODE_ECI = 7
MODE_STRUCTURED_APPEND = 3

# Encoding mode sizes.
MODE_SIZE_SMALL = {
    MODE_NUMBER: 10,
    MODE_ALPHA_NUM: 9,
    MODE_BYTES: 8,
    MODE_KANJI: 8,
}
MODE_SIZE_MEDIUM = {
    MODE_NUMBER: 12,
    MODE_ALPHA_NUM: 11,
    MODE_BYTES: 16,
    MODE_KANJI: 10,
}
MODE_SIZE_LARGE = {
    MODE_NUMBER: 14,
    MODE_ALPHA_NUM: 13,
    MODE_BYTES: 16,
    MODE_KANJI: 12,
}

FORMAT_INFO_MASK = 0b101010000010010
FORMAT_INFO_BCH_GENERATOR = 0b10100110111

ALIGNMENT_POSITIONS = [
    [],
    [6, 18],
    [6, 22],
    [6, 26],
    [6, 30],
    [6, 34],
    [6, 22, 38],
    [6, 24, 42],
    [6, 26, 46],
    [6, 28, 50],
    [6, 30, 54],
    [6, 32, 58],
    [6, 34, 62],
    [6, 26, 46, 66],
    [6, 26, 48, 70],
    [6, 26, 50, 74],
    [6, 30, 54, 78],
    [6, 30, 56, 82],
    [6, 30, 58, 86],
    [6, 34, 62, 90],
    [6, 28, 50, 72, 94],
    [6, 26, 50, 74, 98],
    [6, 30, 54, 78, 102],
    [6, 28, 54, 80, 106],
    [6, 32, 58, 84, 110],
    [6, 30, 58, 86, 114],
    [6, 34, 62, 90, 118],
    [6, 26, 50, 74, 98, 122],
    [6, 30, 54, 78, 102, 126],
    [6, 26, 52, 78, 104, 130],
    [6, 30, 56, 82, 108, 134],
    [6, 34, 60, 86, 112, 138],
    [6, 30, 58, 86, 114, 142],
    [6, 34, 62, 90, 118, 146],
    [6, 30, 54, 78, 102, 126, 150],
    [6, 24, 50, 76, 102, 128, 154],
    [6, 28, 54, 80, 106, 132, 158],
    [6, 32, 58, 84, 110, 136, 162],
    [6, 26, 54, 82, 110, 138, 166],
    [6, 30, 58, 86, 114, 142, 170],
]


DATA_BLOCKS_INFO = [
    # For each version: L, M, Q, H: (EC bytes, block size, blocks count, large blocks count)
    ((7, 19, 1), (10, 16, 1), (13, 13, 1), (17, 9, 1)),  # v1
    ((10, 34, 1), (16, 28, 1), (22, 22, 1), (28, 16, 1)),  # v2
    ((15, 55, 1), (26, 44, 1), (18, 17, 2), (22, 13, 2)),  # v3
    ((20, 80, 1), (18, 32, 2), (26, 24, 2), (16, 9, 4)),  # v4
    ((26, 108, 1), (24, 43, 2), (18, 15, 2, 2), (22, 11, 2, 2)),  # v5
    ((18, 68, 2), (16, 27, 4), (24, 19, 4), (28, 15, 4)),  # v6
    ((20, 78, 2), (18, 31, 4), (18, 14, 2, 4), (26, 13, 4, 1)),  # v7
    ((24, 97, 2), (22, 38, 2, 2), (22, 18, 4, 2), (26, 14, 4, 2)),  # v8
    ((30, 116, 2), (22, 36, 3, 2), (20, 16, 4, 4), (24, 12, 4, 4)),  # v9
    ((18, 68, 2, 2), (26, 43, 4, 1), (24, 19, 6, 2), (28, 15, 6, 2)),  # v10
    ((20, 81, 4), (30, 50, 1, 4), (28, 22, 4, 4), (24, 12, 3, 8)),  # v11
    ((24, 92, 2, 2), (22, 36, 6, 2), (26, 20, 4, 6), (28, 14, 7, 4)),  # v12
    ((26, 107, 4), (22, 37, 8, 1), (24, 20, 8, 4), (22, 11, 12, 4)),  # v13
    ((30, 115, 3, 1), (24, 40, 4, 5), (20, 16, 11, 5), (24, 12, 11, 5)),  # v14
    ((22, 87, 5, 1), (24, 41, 5, 5), (30, 24, 5, 7), (24, 12, 11, 7)),  # v15
    ((24, 98, 5, 1), (28, 45, 7, 3), (24, 19, 15, 2), (30, 15, 3, 13)),  # v16
    ((28, 107, 1, 5), (28, 46, 10, 1), (28, 22, 1, 15), (28, 14, 2, 17)),  # v17
    ((30, 120, 5, 1), (26, 43, 9, 4), (28, 22, 17, 1), (28, 14, 2, 19)),  # v18
    ((28, 113, 3, 4), (26, 44, 3, 11), (26, 21, 17, 4), (26, 13, 9, 16)),  # v19
    ((28, 107, 3, 5), (26, 41, 3, 13), (30, 24, 15, 5), (28, 15, 15, 10)),  # v20
    ((28, 116, 4, 4), (26, 42, 17), (28, 22, 17, 6), (30, 16, 19, 6)),  # v21
    ((28, 111, 2, 7), (28, 46, 17), (30, 24, 7, 16), (24, 13, 34)),  # v22
    ((30, 121, 4, 5), (28, 47, 4, 14), (30, 24, 11, 14), (30, 15, 16, 14)),  # v23
    ((30, 117, 6, 4), (28, 45, 6, 14), (30, 24, 11, 16), (30, 16, 30, 2)),  # v24
    ((26, 106, 8, 4), (28, 47, 8, 13), (30, 24, 7, 22), (30, 15, 22, 13)),  # v25
    ((28, 114, 10, 2), (28, 46, 19, 4), (28, 22, 28, 6), (30, 16, 33, 4)),  # v26
    ((30, 122, 8, 4), (28, 45, 22, 3), (30, 23, 8, 26), (30, 15, 12, 28)),  # v27
    ((30, 117, 3, 10), (28, 45, 3, 23), (30, 24, 4, 31), (30, 15, 11, 31)),  # v28
    ((30, 116, 7, 7), (28, 45, 21, 7), (30, 23, 1, 37), (30, 15, 19, 26)),  # v29
    ((30, 115, 5, 10), (28, 47, 19, 10), (30, 24, 15, 25), (30, 15, 23, 25)),  # v30
    ((30, 115, 13, 3), (28, 46, 2, 29), (30, 24, 42, 1), (30, 15, 23, 28)),  # v31
    ((30, 115, 17), (28, 46, 10, 23), (30, 24, 10, 35), (30, 15, 19, 35)),  # v32
    ((30, 115, 17, 1), (28, 46, 14, 21), (30, 24, 29, 19), (30, 15, 11, 46)),  # v33
    ((30, 115, 13, 6), (28, 46, 14, 23), (30, 24, 44, 7), (30, 16, 59, 1)),  # v34
    ((30, 121, 12, 7), (28, 47, 12, 26), (30, 24, 39, 14), (30, 15, 22, 41)),  # v35
    ((30, 121, 6, 14), (28, 47, 6, 34), (30, 24, 46, 10), (30, 15, 2, 64)),  # v36
    ((30, 122, 17, 4), (28, 46, 29, 14), (30, 24, 49, 10), (30, 15, 24, 46)),  # v37
    ((30, 122, 4, 18), (28, 46, 13, 32), (30, 24, 48, 14), (30, 15, 42, 32)),  # v38
    ((30, 117, 20, 4), (28, 47, 40, 7), (30, 24, 43, 22), (30, 15, 10, 67)),  # v39
    ((30, 118, 19, 6), (28, 47, 18, 31), (30, 24, 34, 34), (30, 15, 20, 61)),  # v40
]

ALPHANUM_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

WHITE = 0
BLACK = 1

class QrReadingException(Exception):
    pass


class QrImageException(QrReadingException):
    pass


class QrCorruptError(QrReadingException):
    pass


class QrImageRecognitionException(QrImageException):
    pass


class QrFormatError(Exception):
    pass


class IllegalQrMessageModeId(QrFormatError):
    def __init__(self, mode_id):
        super(IllegalQrMessageModeId, self).__init__(
            "Unknown mode ID: {0!r:s}".format(
                mode_id,
            )
        )


class IllegalQrVersionError(QrFormatError):
    def __init__(self, version):
        super(IllegalQrVersionError, self).__init__(
            "Illegal QR version: {0!r:s} (should be integer between 1-40)".format(
                version,
            )
        )




# taken from qrcode package
def get_mask_func(mask_id):
    """
    Return the mask function for the given mask pattern.
    :param int mask_id: The mask ID in the range 0-7.
    """
    id_to_mask = {
        0: lambda i, j: (i + j) % 2 == 0,  # 000
        1: lambda i, j: i % 2 == 0,  # 001
        2: lambda i, j: j % 3 == 0,  # 010
        3: lambda i, j: (i + j) % 3 == 0,  # 011
        4: lambda i, j: (i // 2 + j // 3) % 2 == 0,  # 100
        5: lambda i, j: (i * j) % 2 + (i * j) % 3 == 0,  # 101
        6: lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0,  # 110
        7: lambda i, j: ((i * j) % 3 + (i + j) % 2) % 2 == 0,  # 111
    }
    if mask_id in id_to_mask:
        return id_to_mask[mask_id]
    raise QrFormatError("Bad mask pattern: {0!r:s}".format(mask_id))


def mode_sizes_for_version(version):
    if version != int(version):
        raise IllegalQrVersionError(version)
    if 1 <= version <= 9:
        return MODE_SIZE_SMALL
    elif 10 <= version <= 26:
        return MODE_SIZE_MEDIUM
    elif 27 <= version <= 40:
        return MODE_SIZE_LARGE
    raise IllegalQrVersionError(version)


def bits_for_length(version, data_mode):
    size_mode = mode_sizes_for_version(version)

    if data_mode not in size_mode:
        raise QrFormatError(
            "Unknown data type ID: {0!r:s}".format(
                data_mode,
            )
        )

    return size_mode[data_mode]


def size_by_version(version):
    if version < 1 or version > 40 or not version == int(version):
        raise IllegalQrVersionError(version)
    return 17 + version * 4


def ec_level_from_format_info_code(info_ec_code):
    levels = {
        0: ERROR_CORRECT_M,
        1: ERROR_CORRECT_L,
        2: ERROR_CORRECT_H,
        3: ERROR_CORRECT_Q,
    }
    return levels[info_ec_code]


def get_dead_zones(version):
    size = size_by_version(version)
    constant_zones = [
        (0, 0, 8, 8),  # top left position + format-info
        (size - 8, 0, size - 1, 8),  # top right position + format-info
        (0, size - 8, 7, size - 1),  # bottom left position
        (8, size - 7, 8, size - 1),  # bottom left format info
        (8, 6, size - 9, 6),  # top timing array
        (6, 8, 6, size - 9),  # left timing array
    ]

    if version >= 7:
        constant_zones.append((size - 11, 0, size - 9, 5))  # top version info
        constant_zones.append((0, size - 11, 5, size - 9))  # bottom (left) version info

    alignments_zones = []
    alignment_centers = list(permutations(ALIGNMENT_POSITIONS[version - 1], 2))
    alignment_centers.extend((x, x) for x in ALIGNMENT_POSITIONS[version - 1])

    for center_x, center_y in alignment_centers:
        alignment_zone = (center_x - 2, center_y - 2, center_x + 2, center_y + 2)
        if all(
            not is_rect_overlapping(alignment_zone, dead_zone)
            for dead_zone in constant_zones
        ):
            alignments_zones.append(alignment_zone)
    return constant_zones + alignments_zones


class QRDecoder(object):
    def __init__(self, scanner):
        self.scanner = scanner

    @property
    def version(self):
        return self.scanner.info.version

    def get_first(self):
        return self._decode_next_message()

    def __iter__(self):
        yield self._decode_next_message()

    def get_all(self):
        return list(self)

    def _decode_next_message(self):
        mode = self.scanner.read_int(4)
        return self._decode_message(mode)

    def _decode_message(self, mode):
        if mode == MODE_NUMBER:
            message = self._decode_numeric_message()
        elif mode == MODE_ALPHA_NUM:
            message = self._decode_alpha_num_message()
        elif mode == MODE_BYTES:
            message = self._decode_bytes_message()
        elif mode == MODE_KANJI:
            message = self._decode_kanji_message()
        elif mode == MODE_STRUCTURED_APPEND:
            raise NotImplementedError("Structured append encoding not implemented yet")
        elif mode == MODE_ECI:
            raise NotImplementedError(
                "Extended Channel Interpretation encoding not implemented yet"
            )
        else:
            raise IllegalQrMessageModeId(mode)
        return message

    def _decode_numeric_message(self):
        char_count = self.scanner.read_int(bits_for_length(self.version, MODE_NUMBER))
        val = 0
        triples, rest = divmod(char_count, 3)
        for _ in range(triples):
            val = val * 1000 + self.scanner.read_int(10)
        if rest == 2:
            val = val * 100 + self.scanner.read_int(7)
        elif rest == 1:
            val = val * 10 + self.scanner.read_int(4)

        return val

    def _decode_alpha_num_message(self):
        char_count = self.scanner.read_int(
            bits_for_length(self.version, MODE_ALPHA_NUM)
        )
        val = ""
        doubles, has_single = divmod(char_count, 2)
        for _ in range(doubles):
            double = self.scanner.read_int(11)
            val += ALPHANUM_CHARS[double // 45] + ALPHANUM_CHARS[double % 45]
        if has_single:
            val += ALPHANUM_CHARS[self.scanner.read_int(6)]
        return val

    def _decode_bytes_message(self):
        char_count = self.scanner.read_int(bits_for_length(self.version, MODE_BYTES))
        raw = ints_to_bytes(self.scanner.read_int(8) for _ in range(char_count))
        try:
            val = raw.decode("utf-8")
        except UnicodeDecodeError:
            val = raw.decode("iso-8859-1")
        return val

    def _decode_kanji_message(self):
        char_count = self.scanner.read_int(bits_for_length(self.version, MODE_KANJI))
        nums = []
        for _ in range(char_count):
            mashed = self.scanner.read_int(13)
            num = ((mashed // 0xC0) << 8) + mashed % 0xC0
            num += 0x8140 if num < 0x1F00 else 0xC140
            nums.extend(divmod(num, 2**8))
        return ints_to_bytes(nums).decode("shift-jis")


def format_info_check(format_info):
    """Returns 0 if given a complete format info code and it is valid.
    Otherwise, returns a positive number.
    If given a format info marker padded with 10 bits (e.g. 101010000000000) returns
    the corresponding 10-bit error correction code to append to it
    :param int format_info: The format info with error correction (15 bits) or without it (5 bits)
    :rtype: int
    """
    g = FORMAT_INFO_BCH_GENERATOR
    for i in range(4, -1, -1):
        if format_info & (1 << (i + 10)):
            format_info ^= g << i
    return format_info


def hamming_diff(a, b):
    """Calculates the hamming weight of the difference between two number (number of different bits)
    :param int a: A number to calculate the diff from
    :param int b: A number to calculate the diff from
    :return: The amount of different bits
    :rtype: int
    """
    weight = 0
    diff = a ^ b
    while diff > 0:
        weight += diff & 1
        diff >>= 1
    return weight


def validate_format_info(format_info, second_format_info_sample=None):
    """
    Receives one or two copies of a QR format info containing error correction bits, and returns just the format
    info bits, after error checking and correction
    :param int format_info: The 15-bit format info bits with the error correction info
    :param int second_format_info_sample: The secondary 15-bit format info bits with the error correction info
    :raise QrCorruptError: in case the format info is too corrupt to singularly verify
    :return: The 5-bit (0-31) format info number
    :rtype: int
    """
    if second_format_info_sample is None:
        second_format_info_sample = format_info
    if (
        format_info_check(format_info)
        == format_info_check(second_format_info_sample)
        == 0
    ):
        return format_info >> 10
    format_info = (format_info << 15) + second_format_info_sample

    best_format = None
    max_distance = 29
    for test_format in range(0, 32):
        test_code = (test_format << 10) ^ format_info_check(test_format << 10)
        test_dist = hamming_diff(format_info, test_code + (test_code << 15))

        if test_dist < max_distance:
            max_distance = test_dist
            best_format = test_format
        elif test_dist == max_distance:
            best_format = None
    if best_format is None:
        raise QrCorruptError("QR meta-info is too corrupt to read")
    return best_format


def validate_data(data, version, ec_level):
    return data


def add(t1, t_or_n):
    if isinstance(t_or_n, (int, float)):
        return tuple(x + t_or_n for x in t1)
    elif isinstance(t_or_n, tuple):
        return tuple(x + t_or_n[i] for i, x in enumerate(t1))
    raise TypeError("Can't add a %s to a tuple" % type(t_or_n))


def multiply(t1, t_or_n):
    if isinstance(t_or_n, (int, float)):
        return tuple(x * t_or_n for x in t1)
    elif isinstance(t_or_n, tuple):
        return tuple(x * t_or_n[i] for i, x in enumerate(t1))
    raise TypeError("Can't multiply a tuple by a" % type(t_or_n))




class Scanner(object):
    def __init__(self):
        self._current_index = -1
        self._data_len = 0
        self._info = None
        self.data = None
        self._was_read = False

    @property
    def info(self):
        """The meta info for the QR code. Reads the code on access if needed.
        :rtype: QRCodeInfo
        """
        if not self._was_read:
            self.read()
        return self._info

    def read(self):
        self._was_read = True
        self.read_info()
        self.data = validate_data(
            self._read_all_data(), self.info.version, self.info.error_correction_level
        )
        self._data_len = len(self.data)
        self.reset()

    def read_info(self):
        raise NotImplementedError()

    def _read_all_data(self):
        raise NotImplementedError()

    # Iteration methods #

    def reset(self):
        self._current_index = -1

    def read_bit(self):
        if not self._was_read:
            self.read()
        self._current_index += 1
        if self._current_index >= self._data_len:
            self._current_index = self._data_len
            raise StopIteration()
        return self.data[self._current_index]

    def read_int(self, amount_of_bits):
        if not self._was_read:
            self.read()
        val = 0
        bits = [self.read_bit() for _ in range(amount_of_bits)]
        for bit in bits:
            val = (val << 1) + bit
        return val

    def __iter__(self):
        while True:
            try:
                yield self.read_bit()
            except StopIteration:
                return


class ImageScanner(Scanner):
    def __init__(self, image):
        """
        :type image: PIL.Image.Image
        :return:
        """
        super(ImageScanner, self).__init__()
        # self.image = image.convert('LA')  # gray-scale it baby!
        self.image = image
        self.mask = None

    def get_mask(self):
        mask_func = get_mask_func(self.info.mask_id)
        return {
            (x, y): 1 if mask_func(y, x) else 0
            for x in range(self.info.size)
            for y in range(self.info.size)
        }

    def read_info(self):
        info = QRCodeInfo()
        info.canvas = self.get_image_borders()
        info.block_size = self.get_block_size(info.canvas[:2])
        info.size = int((info.canvas[2] - (info.canvas[0]) + 1) / info.block_size[0])
        info.version = (info.size - 17) // 4
        self._info = info
        self._read_format_info()
        self.mask = self.get_mask()
        return info

    def _get_pixel(self, coords):
        try:
            shade, alpha = self.image.getpixel(coords)
            return BLACK if shade < 128 and alpha > 0 else WHITE
        except IndexError:
            return WHITE

    def get_image_borders(self):
        def get_corner_pixel(canvas_corner, vector, max_distance):
            for dist in range(max_distance):
                for x in range(dist + 1):
                    coords = (
                        canvas_corner[0] + vector[0] * x,
                        canvas_corner[1] + vector[1] * (dist - x),
                    )
                    if self._get_pixel(coords) == BLACK:
                        return coords
            raise QrImageRecognitionException(
                "Couldn't find one of the edges ({0:s}-{1:s})".format(
                    ("top", "bottom")[vector[1] == -1],
                    ("left", "right")[vector[0] == -1],
                )
            )

        max_dist = min(self.image.width, self.image.height)
        min_x, min_y = get_corner_pixel((0, 0), (1, 1), max_dist)
        max_x, max_x_y = get_corner_pixel((self.image.width - 1, 0), (-1, 1), max_dist)
        max_y_x, max_y = get_corner_pixel((0, self.image.height - 1), (1, -1), max_dist)
        if max_x_y != min_y:
            raise QrImageRecognitionException(
                "Top-left position pattern not aligned with the top-right one"
            )
        if max_y_x != min_x:
            raise QrImageRecognitionException(
                "Top-left position pattern not aligned with the bottom-left one"
            )
        return min_x, min_y, max_x, max_y

    def get_block_size(self, img_start):
        """
        Returns the size in pixels of a single block.
        :param tuple[int, int] img_start: The topmost left pixel in the QR (MUST be black or dark).
        :return: A tuple of width, height in pixels of a block
        :rtype: tuple[int, int]
        """
        pattern_size = 7

        left, top = img_start
        block_height, block_width = None, None
        for i in range(1, (self.image.width - left) // pattern_size):
            if self._get_pixel((left + i * pattern_size, top)) == WHITE:
                block_width = i
                break
        for i in range(1, (self.image.height - top) // pattern_size):
            if self._get_pixel((left, top + i * pattern_size)) == WHITE:
                block_height = i
                break
        return block_width, block_height

    def _read_format_info(self):
        source_1 = (
            self._get_straight_bits((8, -7), 7, "d") << 8
        ) + self._get_straight_bits((-1, 8), 8, "l")
        source_2 = (
            self._get_straight_bits((7, 8), 8, "l", (1,)) << 8
        ) + self._get_straight_bits((8, 0), 9, "d", (6,))

        format_info = validate_format_info(
            source_1 ^ FORMAT_INFO_MASK, source_2 ^ FORMAT_INFO_MASK
        )
        self.info.error_correction_level = ec_level_from_format_info_code(
            format_info >> 3
        )
        self.info.mask_id = format_info & 0b111

    def _read_all_data(self):
        pos_iterator = QrZigZagIterator(
            self.info.size, get_dead_zones(self.info.version)
        )
        return [self._get_bit(pos) ^ self.mask[pos] for pos in pos_iterator]

    def _get_bit(self, coords):
        x, y = coords
        if x < 0:
            x += self.info.size
        if y < 0:
            y += self.info.size
        return self._get_pixel(
            add(self.info.canvas[:2], multiply((x, y), self.info.block_size))
        )

    def _get_straight_bits(self, start, length, direction, skip=()):
        """
        Reads several bits from the specified coordinates
        :param tuple[int] start: The x, y of the start position
        :param int length: the amount of bits to read
        :param str direction: d(own) or l(eft)
        :param tuple skip: the indexes to skip. they will still be counted on for the length
        :return: The bits read as an integer
        :rtype: int
        """
        result = 0
        counted = 0
        step = (0, 1) if direction == "d" else (-1, 0)
        for i in range(length):
            if i in skip:
                start = add(start, step)
                continue
            result += self._get_bit(start) << counted
            counted += 1
            start = add(start, step)
        return result


class QrZigZagIterator(Iterator):
    def __init__(self, size, dead_zones):
        self.size = size
        self.ignored_pos = {
            (x, y)
            for zone in dead_zones
            for x in range(zone[0], zone[2] + 1)
            for y in range(zone[1], zone[3] + 1)
        }
        self._current = ()
        self._scan_direction = "u"
        self._odd_col_modifier = False
        self.reset()

    def reset(self):
        self._current = (self.size - 2, self.size)
        self._scan_direction = "u"
        self._odd_col_modifier = False

    def _advance_pos(self):
        pos = self._current
        while pos[0] >= 0 and (pos == self._current or pos in self.ignored_pos):
            step = (-1, 0)
            # We advance a line if we're in an odd column, but if we have the col_modified flag on, we switch it around
            advance_line = ((self.size - pos[0]) % 2 == 0) ^ self._odd_col_modifier
            if advance_line:
                step = (1, -1 if self._scan_direction == "u" else 1)
                # if we're trying to advance a line but we've reached the edge, we should change directions
                if (pos[1] == 0 and self._scan_direction == "u") or (
                    pos[1] == self.size - 1 and self._scan_direction == "d"
                ):
                    # swap scan direction
                    self._scan_direction = "d" if self._scan_direction == "u" else "u"
                    # go one step left
                    step = (-1, 0)
                    # make sure we're not tripping over the timing array
                    if pos[0] > 0 and all(
                        (pos[0] - 1, y) in self.ignored_pos for y in range(self.size)
                    ):
                        step = (-2, 0)
                        self._odd_col_modifier = not self._odd_col_modifier
            pos = add(pos, step)
        self._current = pos

    def __next__(self):
        self._advance_pos()
        if self._current[0] < 0:
            raise StopIteration()
        return self._current

    next = __next__


class QRCodeInfo(object):
    # number between 1-40
    version = 0

    # the error correction level.
    error_correction_level = 0

    # the id of the mask (0-7)
    mask_id = 0

    # the part of the image that contains the QR code
    canvas = (0, 0)

    # the size of each block in pixels
    block_size = (0, 0)

    # the amount of blocks at each side of the image (it's always a square)
    size = 0

    def __str__(self):
        return "<version %s, ec %s, mask %s>" % (
            self.version,
            self.error_correction_level,
            self.mask_id,
        )


class PILSubstitute:
    def __init__(self, mat):
        self.mat = mat
        self.width = len(mat[0])
        self.height = len(mat)

    def getpixel(self, coords):
        # should returns shade, alpha
        # mat is binary (but this works if it's positive for black, non-positive for white)
        if self.mat[coords[1]][coords[0]] > 0:
            return 0, 255
        else:
            return 255, 255

    def convert(self, mode):
        return self


if __name__ == "__main__":
    N = int(input())
    mat = []
    for _ in range(N):
        row = input()
        mat.append([int(c) for c in row])

    # Create a file-like object
    file_like_object = PILSubstitute(mat)

    data = ImageScanner(file_like_object)
    result = QRDecoder(data).get_first()
    print(result)
