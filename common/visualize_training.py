import visdom
import numpy as np
import matplotlib.pyplot as plt


class TrainingStats(object):

    def __init__(self, num_classes=10):
        self.losses = []
        self.train_acc = []
        self.val_acc = []
        self.iters = []
        self.connected = True

        # Try to connect to the server.
        try:
            self.vis = visdom.Visdom(raise_exceptions=True)
            print("Visdom is active.")
        except Exception as e:
            self.connected = False
            print("Warning: No connection to the server can be established. "
                  "Visualizations will not be shown.")

        if not self.connected:
            return

        self.vis.close()

        self.loss_window = self.vis.scatter(
            X=[[0, 0]],
            opts=dict(
                title='Training loss',
                xmin=0,
                ymin=0,
                xlabel='Iterations',
                ylabel='Loss'
            ),
        )

        self.train_acc_window = self.vis.scatter(
            X=[[0, 0]],
            opts=dict(
                title='Training accuracy',
                xmin=0,
                ymin=0,
                ymax=100,
                xlabel='Iterations',
                ylabel='Train accuracy'
            ),
        )

        self.val_acc_window = self.vis.scatter(
            X=[[0, 0]],
            opts=dict(
                title='Validation accuracy',
                xmin=0,
                ymin=0,
                ymax=100,
                xlabel='Iterations',
                ylabel='Val accuracy'
            ),
        )

        self.heatmap_opts = dict(colormap="Jet", xmin=0, xmax=num_classes - 1)

        empty_img = np.zeros((3, 480, 640))
        # self.prediction_window = self.vis.heatmap(
        #     np.zeros((480, 640)), opts=self.heatmap_opts
        # )
        # self.mask_window = self.vis.heatmap(
        #     np.zeros((480, 640)), opts=self.heatmap_opts
        # )
        self.img_window = self.vis.image(
            empty_img, opts=dict(caption="Original Image")
        )
        self.prediction_window = self.vis.image(
            empty_img, opts=dict(caption="Predictions")
        )
        self.mask_window = self.vis.image(
            empty_img, opts=dict(caption="Ground truth mask")
        )

        # self.prediction_window = None
        # self.mask_window = None
        # self.img_window = None
        self.best_prediction_window = None
        self.best_mask_window = None
        self.worst_prediction_window = None
        self.worst_mask_window = None
        self.imgs_window = None

    def is_connected(self):
        return self.connected and self.vis.check_connection(timeout_seconds=3)

    def show_args(self, args):
        if not self.is_connected():
            return

        # Show args.
        print("show")
        print(vars(args))
        # self.vis.properties([vars(args)])
        # exit()

    def update(self, iteration, loss, train_acc, val_acc=None,
               predictions=None, mask=None, img=None):
        """Updates the visualizer.

        Arguments:
            iteration {int} -- The training iteration.
            loss {float} -- The loss at this training iteration.
            train_acc {loss} -- The training accuracy at this training iteration.

        Keyword Arguments:
            val_acc {float} -- The validation accuracy at this iteration. (default: {None})
            predictions {np.ndarray, [(2/3x)HxW]} -- The predicted mask for this class.
                If tensor is [HxW], each pixel contains the semantic label at that pixel.
                If tensor is [2xHxW], the mask at predictions[0] is the best prediction in
                the batch and the mask at predictions[1] is the worst. If tensor is [3xHxW],
                then the mask is treated as an RGB image. (default: {None})
            mask {np.ndarray, [(3x)HxW]} -- The ground truth mask for the image. If tensor
                is [HxW], each pixel contains the semantic label at that pixel. If tensor
                is [3xHxW], then the mask is treated as an RGB image. (default: {None})
            img {np.ndarray, [3xHxW]} -- The RGB image. (default: {None})
        """
        if not self.is_connected():
            return

        self.losses.append(loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.iters.append(iteration)

        self.vis.scatter(X=[[iteration, loss]], win=self.loss_window,
                         update='append')

        self.vis.scatter(X=[[iteration, train_acc]], win=self.train_acc_window,
                         update='append')

        if val_acc is not None:
            self.vis.scatter(X=[[iteration, val_acc]], win=self.val_acc_window,
                             update='append')

        if predictions is not None:
            # Run checks.
            assert predictions.ndim in [2, 3], "Prediction must have dim 2 or 3 but has {}".format(predictions.shape)
            assert mask.ndim == predictions.ndim, "Prediction and mask have different dimensions."
            assert predictions.shape[0] in [2, 3], "Prediction must be [2/3xHxW] but has {}".format(predictions.shape)

            if predictions.ndim == 2 or predictions.shape[0] == 3:
                self.show_img(iteration, predictions, mask, img)
            elif predictions.ndim == 3:
                self._show_best_worst(iteration, predictions, mask, img)

    def show_img(self, iteration, predictions, mask, img=None):
        if not self.is_connected():
            return

        if predictions.ndim == 2:
            self.vis.heatmap(
                np.flip(predictions, axis=0), win=self.prediction_window, opts=self.heatmap_opts
            )

            self.vis.heatmap(
                np.flip(mask, axis=0), win=self.mask_window, opts=self.heatmap_opts
            )
        else:
            self.vis.image(
                predictions, win=self.prediction_window, opts=dict(caption="Predictions")
            )

            self.vis.image(
                mask, win=self.mask_window, opts=dict(caption="Ground truth mask")
            )

        if img is not None:
            self.vis.image(
                img, opts=dict(caption="Original Image"), win=self.img_window
            )

    def _show_best_worst(self, iteration, predictions, masks, imgs=None, scores=None):
        self.best_prediction_window = self.vis.heatmap(
            np.flip(predictions[0, :, :], axis=0), win=self.best_prediction_window,
            opts={**self.heatmap_opts, 'title': "Best prediction"}
        )

        self.worst_prediction_window = self.vis.heatmap(
            np.flip(predictions[1, :, :], axis=0), win=self.worst_prediction_window,
            opts={**self.heatmap_opts, 'title': "Worst prediction"}
        )

        self.best_mask_window = self.vis.heatmap(
            np.flip(masks[0, :, :], axis=0), win=self.best_mask_window,
            opts={**self.heatmap_opts, 'title': "Best mask"}
        )

        self.worst_mask_window = self.vis.heatmap(
            np.flip(masks[1, :, :], axis=0), win=self.worst_mask_window,
            opts={**self.heatmap_opts, 'title': "Worst mask"}
        )

        if imgs is not None:
            self.imgs_window = self.vis.images(
                imgs, opts=dict(caption="Original Normalized Images, Best and Worst"), win=self.imgs_window
            )
