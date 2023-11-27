import numpy as np


class DataProcessor:
    def __init__(self, num_workers=8):
        self.num_workers = num_workers

    def vectorize_data(self, data):
        data["curve"]["x"] = np.array(data["curve"]["x"], dtype=np.float32)
        data["curve"]["y"] = np.array(data["curve"]["y"], dtype=np.float32)

    def normalize_data(self, data):
        width = data["curve"]["grid"]["width"]
        height = data["curve"]["grid"]["height"]

        data["curve"]["x"] /= width
        data["curve"]["y"] /= height

        for k in data["curve"]["grid"]["keys"]:
            k["hitbox"]["x"] /= width
            k["hitbox"]["y"] /= height
            k["hitbox"]["w"] /= width
            k["hitbox"]["h"] /= height

        data["curve"]["grid"]["width"] = 1.0
        data["curve"]["grid"]["height"] = 1.0

    def extrapolate_segment_curve(self, x, n_parts=300):
        n = len(x)
        add_parts = n_parts - n
        x_new = []

        for i in range(n):
            x_new.append(x[i])
            if i == n - 1:
                break
            cnt = add_parts // (n - 1) + (i < (add_parts % (n - 1)))
            for j in range(1, cnt + 1):
                ratio = j / (cnt + 1)
                x_new.append((1 - ratio) * x[i] + ratio * x[i + 1])
        assert len(x_new) == n_parts or n == 1
        while len(x_new) != n_parts:
            x_new.append(x[0])
        return np.array(x_new)

    def compress_curve(self, x, n_points=100):
        ans = []
        for i in range(n_points):
            idx = round((i / (n_points - 1)) * (x.shape[0] - 1))
            ans.append(x[idx])
        return np.array(ans)

    def get_processed_item(self, data):
        """
        Returns input, target for the network.
        """
        self.vectorize_data(data)
        self.normalize_data(data)

        data["curve"]["x"] = self.extrapolate_segment_curve(data["curve"]["x"])
        data["curve"]["y"] = self.extrapolate_segment_curve(data["curve"]["y"])
        data["curve"]["x"] = self.compress_curve(data["curve"]["x"])
        data["curve"]["y"] = self.compress_curve(data["curve"]["y"])

        return np.stack([data["curve"]["x"], data["curve"]["y"]], axis=0)

    def get_char_coords(self, data, c):
        for k in data["curve"]["grid"]["keys"]:
            if k.get("label", None) == c:
                x = k["hitbox"]["x"] + k["hitbox"]["w"] / 2
                y = k["hitbox"]["y"] + k["hitbox"]["h"] / 2
                return x, y
        if c == "-":
            return None, None
        if c == "ъ":
            return self._get_char_coords(data, "ь")
        if c == "ё":
            return self._get_char_coords(data, "е")

    def _bin_search_x(self, voc, x):
        l = 0
        r = len(voc)
        while r - l > 1:
            m = (l + r) // 2
            if voc[m][1]["x_begin"] < x:
                l = m
            else:
                r = m
        return l
