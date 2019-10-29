from typing import List

import numpy as np
from sklearn.preprocessing import MinMaxScaler, normalize


def diagram_preprocess(diagrams, scalers):
    if len(diagrams) == 1:
        P = diagrams[0]
    else:
        P = np.concatenate(diagrams, 0)
    for indices, scaler in scalers:
        scaler.fit(P[:, indices])

    diagrams = [np.copy(d) for d in diagrams]
    for i in range(len(diagrams)):
        if diagrams[i].shape[0] > 0:
            for (indices, scaler) in scalers:
                diagrams[i][:, indices] = scaler.transform(diagrams[i][:, indices])
    return diagrams, scalers


def persistence_image(
        diagrams: List[np.ndarray],
        bandwidth=1.,
        weight_fn=lambda x: 1,
        resolution=(20, 20),
        im_range=(np.nan, np.nan, np.nan, np.nan)):
    if np.isnan(np.array(im_range)).any():
        diagrams, scalers = diagram_preprocess(diagrams, scalers=[([0, 1], MinMaxScaler())])
        [mx, my], [Mx, My] = scalers[0][1].data_min_, scalers[0][1].data_max_
        im_range = np.where(np.isnan(np.array(im_range)), np.array([mx, Mx, my, My]), np.array(im_range))

    num_diag, Xfit = len(diagrams), []
    for i in range(num_diag):
        diagram, num_pts_in_diag = diagrams[i], diagrams[i].shape[0]
        w = np.ones(num_pts_in_diag)
        for j in range(num_pts_in_diag):
            w[j] = weight_fn(diagram[j, :])

        x_values = np.linspace(im_range[0], im_range[1], resolution[0])
        y_values = np.linspace(im_range[2], im_range[3], resolution[1])

        Xs = np.tile(
            (diagram[:, 0][:, np.newaxis, np.newaxis] - x_values[np.newaxis, np.newaxis, :]),
            [1, resolution[1], 1])
        Ys = np.tile(
            diagram[:, 1][:, np.newaxis, np.newaxis] - y_values[np.newaxis, :, np.newaxis],
            [1, 1, resolution[0]])
        image = np.tensordot(w, np.exp((-np.square(Xs) - np.square(Ys)) / (2 * np.square(bandwidth))) / (
                    bandwidth * np.sqrt(2 * np.pi)), 1)

        Xfit.append(image.flatten()[np.newaxis, :])
        # Xfit = np.concatenate(Xfit, 0)

    Xfit = np.array(Xfit).reshape(len(diagrams), -1)
    Xfit = normalize(Xfit)
    print(Xfit)
    return Xfit


def persistence_landscape(
        dg: List[np.ndarray],
        num_landscapes=5,
        resolution=100,
        ls_range=(np.nan, np.nan)):
    if np.isnan(np.array(ls_range)).any():
        dg, scalers = diagram_preprocess(dg, scalers=[([0, 1], MinMaxScaler())])
        [mx, my], [Mx, My] = scalers[0][1].data_min_, scalers[0][1].data_max_
        ls_range = np.where(np.isnan(np.array(ls_range)), np.array([mx, My]), np.array(ls_range))

    num_diag, Xfit = len(dg), []
    x_values = np.linspace(ls_range[0], ls_range[1], resolution)
    step_x = x_values[1] - x_values[0]

    for i in range(num_diag):
        diagram = dg[i]

        ls = np.zeros([num_landscapes, resolution])

        events = []
        for j in range(resolution):
            events.append([])

        for j in range(len(dg[i])):
            [px, py] = diagram[j, :]
            min_idx = np.minimum(
                np.maximum(np.ceil((px - ls_range[0]) / step_x).astype(int), 0),
                resolution)
            mid_idx = np.minimum(
                np.maximum(np.ceil((0.5 * (py + px) - ls_range[0]) / step_x).astype(int), 0),
                resolution)
            max_idx = np.minimum(
                np.maximum(np.ceil((py - ls_range[0]) / step_x).astype(int), 0),
                resolution)

            if min_idx < resolution and max_idx > 0:

                landscape_value = ls_range[0] + min_idx * step_x - px
                for k in range(min_idx, mid_idx):
                    events[k].append(landscape_value)
                    landscape_value += step_x

                landscape_value = py - ls_range[0] - mid_idx * step_x
                for k in range(mid_idx, max_idx):
                    events[k].append(landscape_value)
                    landscape_value -= step_x

        for j in range(resolution):
            events[j].sort(reverse=True)
            for k in range(min(num_landscapes, len(events[j]))):
                ls[k, j] = events[j][k]

        Xfit.append(np.sqrt(2) * np.reshape(ls, [1, -1]))

        # Xfit = np.concatenate(Xfit, 0)

    Xfit = np.array(Xfit).reshape(len(dg), -1)

    return Xfit
