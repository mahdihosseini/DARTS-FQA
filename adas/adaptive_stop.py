import sys
import numpy as np

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .metrics import Metrics
else:
    from optim.metrics import Metrics

# conv_start_id_normal = [3, 89, 278, 364, 553, 639]
# conv_start_id_reduce = [175, 450]
# num_conv_per_edge_normal = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
# num_conv_per_edge_reduce = [8, 8, 8, 8, 6, 8, 8, 6, 6, 8, 8, 6, 6, 6]

class StopChecker:

    def __init__(self, args) -> None:
        self.smooth_window_size = 5
        self.epsilon = 0
        self.conv_layers_index_stop = None
        self.node = args.node
        self.layers = args.layers
        self.num_normal_cell_stop = args.num_normal_cell_stop
        self.num_reduce_cell_stop = args.num_reduce_cell_stop
        self._initialize()

    def _initialize(self):
        self.num_conv_per_edge_normal = []
        self.num_conv_per_edge_reduce = []

        for i in range(self.node):
            for j in range(i+2):
                self.num_conv_per_edge_normal.append(12)
                if j < 2:
                    self.num_conv_per_edge_reduce.append(14)
                else:
                    self.num_conv_per_edge_reduce.append(12)
        total_conv_normal = np.sum(self.num_conv_per_edge_normal)
        total_conv_reduce = np.sum(self.num_conv_per_edge_reduce)

        start_id = 3
        self.conv_start_id_normal = []
        self.conv_start_id_reduce = []
        for i in range(self.layers):
            if i in [self.layers // 3, 2 * self.layers // 3]:
                self.conv_start_id_reduce.append(start_id)
                start_id += total_conv_reduce
                start_id += 3
            else:
                self.conv_start_id_normal.append(start_id)
                start_id += total_conv_normal
                start_id += 2

        self.normal_edge_index_stop = np.zeros(len(self.num_conv_per_edge_normal), dtype=bool)  # 14
        self.reduce_edge_index_stop = np.zeros(len(self.num_conv_per_edge_reduce), dtype=bool)

    def local_stop(self, metrics: Metrics, epoch: int) -> None:
        """Use the smoothed input knowledge gain to determine whether to stop
        searching on a certain edge.

        :param metrics:
        :param epoch:
        :return: None

        For each edge, if any of its average delta knowledge gain in each cell is
        below a threshold, we stop searching this edge.
        """
        # take last {window_size+1} cells for smoothing
        S_replica = np.tile(
            A=metrics.KG(0),
            reps=(0, 1))
        for iteration in range(self.smooth_window_size + 2):
            epoch_identifier = (epoch - self.smooth_window_size - 1 + iteration)
            S_replica = np.concatenate((
                S_replica,
                np.tile(
                    A=metrics.KG(epoch_identifier),
                    reps=(1, 1))))

        # padding for smoothing
        S_replica = np.pad(S_replica, ((0, 2), (0, 0)), 'edge')

        # smoothing
        num_layers = S_replica.shape[1]
        for layer in range(num_layers):
            S_replica[:, layer] = np.convolve(S_replica[:, layer],
                np.ones(self.smooth_window_size) / self.smooth_window_size, mode='same')

        if self.conv_layers_index_stop is None:
            self.conv_layers_index_stop = np.zeros(S_replica.shape[1], dtype=bool)

        # since we repeat (padding) the last S for smoothing, now we remove it
        S_replica = S_replica[:-2, :]

        # adaptive-stopping criterion: knowledge gain
        conv_layers_delta_S = S_replica[-1, :] - S_replica[-self.smooth_window_size, :]
        # conv_layers_index_stop = np.zeros_like(conv_layers_delta_S, dtype=bool)

        # edges in normal cell
        offset_start = 0
        offset_end = 0
        # Iterate through 14 edges in normal cells
        for edge in range(len(self.num_conv_per_edge_normal)):

            # check if the current edge is already stopped
            if self.normal_edge_index_stop[edge]:
                offset_end += self.num_conv_per_edge_normal[edge]
                offset_start += self.num_conv_per_edge_normal[edge]
                continue

            offset_end += self.num_conv_per_edge_normal[edge]
            # stop_flag = False
            stop_flag_cell = np.zeros(len(self.conv_start_id_normal), dtype=bool)

            # Compute the average of all conv layers in each edge and each cell
            for cell in range(len(self.conv_start_id_normal)):
                start = self.conv_start_id_normal[cell] + offset_start
                end = self.conv_start_id_normal[cell] + offset_end

                avg_delta_S = np.mean(conv_layers_delta_S[start: end])
                # conv_layers_delta_S[start: end] = avg_delta_S

                # If any avg_delta_S is below the threshold, stop_flag will be True
                # stop_flag = stop_flag | (avg_delta_S < self.epsilon)
                stop_flag_cell[cell] = (avg_delta_S < self.epsilon)

            # For each edge, if its delta_S in more than ${self.num_normal_cell_stop} 
            # cells are below a threshold, stop searching it
            stop_flag = np.sum(stop_flag_cell) >= self.num_normal_cell_stop
            self.normal_edge_index_stop[edge] = stop_flag
            for cell in range(len(self.conv_start_id_normal)):
                start = self.conv_start_id_normal[cell] + offset_start
                end = self.conv_start_id_normal[cell] + offset_end

                self.conv_layers_index_stop[start: end] = stop_flag

            offset_start += self.num_conv_per_edge_normal[edge]

        # edges in reduction cell
        offset_start = 0
        offset_end = 0
        # Iterate through 14 edges in reduction cells
        for edge in range(len(self.num_conv_per_edge_reduce)):

            # check if the current edge is already stopped
            if self.reduce_edge_index_stop[edge]:
                offset_end += self.num_conv_per_edge_reduce[edge]
                offset_start += self.num_conv_per_edge_reduce[edge]
                continue

            offset_end += self.num_conv_per_edge_reduce[edge]
            # stop_flag = False
            stop_flag_cell = np.zeros(len(self.conv_start_id_normal), dtype=bool)

            # Compute the average of all conv layers in each edge and each cell
            for cell in range(len(self.conv_start_id_reduce)):
                start = self.conv_start_id_reduce[cell] + offset_start
                end = self.conv_start_id_reduce[cell] + offset_end

                avg_delta_S = np.mean(conv_layers_delta_S[start: end])
                # conv_layers_delta_S[start: end] = avg_delta_S

                # If any avg_delta_S is below the threshold, stop_flag will be True
                # stop_flag = stop_flag | (avg_delta_S < self.epsilon)
                stop_flag_cell[cell] = (avg_delta_S < self.epsilon)

            # For each edge, if its delta_S in more than ${self.num_reduce_cell_stop} 
            # is below a threshold, stop searching it
            stop_flag = np.sum(stop_flag_cell) >= self.num_reduce_cell_stop
            self.reduce_edge_index_stop[edge] = stop_flag
            for cell in range(len(self.conv_start_id_reduce)):
                start = self.conv_start_id_reduce[cell] + offset_start
                end = self.conv_start_id_reduce[cell] + offset_end

                self.conv_layers_index_stop[start: end] = stop_flag

            offset_start += self.num_conv_per_edge_reduce[edge]

        # Update layers_index_todo
        conv_index = 0
        for layer_index in range(len(metrics.layers_index_todo)):
            if layer_index in metrics.mask:
                continue
            else:
                metrics.layers_index_todo[layer_index] = ~self.conv_layers_index_stop[conv_index]
                conv_index += 1

    def global_stop(self, H) -> None:
        pass







