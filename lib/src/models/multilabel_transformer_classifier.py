import torch
import argparse
import wandb
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict
from pytorch_lightning import LightningDataModule
from lib.src.models.base_transformer_classifier import TransformerClassifier
from collections import OrderedDict

#TODO - inherit from transformer as well as multilabel base
class MultiLabelTransformerClassifier(TransformerClassifier):
    """
    Sample model to show how to use a Transformer model to classify sentences.

    This model builds upon the text representation class then attaches a head from the fully connected classifier class,
    stitching them together according to your config params

    :param args: ArgumentParser containing the hyperparameters.
    :param data: LightningDataModule object containing implementations of train_dataloader, val_dataloader,
     and necessary other ETL.
    """

    def __init__(self, args: argparse.Namespace, data: LightningDataModule) -> None:
        super().__init__(args, data)

    def _get_class_weights(self):
        """
        For the multilabel loss function, we still have sparse labels, it's just they are treated
        independently. So in this case, we just have a list of values, and for each value, there
        is a weight for the positives that is different. We will pass this list of positives to
        the BCEWithLogitsLoss function.

        For more info, see:
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        """
        return self.data.pos_weights

    def _build_loss(self):
        """ Initializes the loss function/s."""
        # have to do it this way to preserve label order
        pos_weights = [self.data.pos_weights[k]
                       for k in self.args.hparams["label"]]
        pos_weights = torch.FloatTensor(pos_weights)
        self._loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # TODO - maybe eliminate this. seems not to be called
    # TODO: allow batch prediction?
    def predict(self, data_module: LightningDataModule, sample: dict) -> dict:
        """ Predict function.
        Args:
            data_module: module with method prepare_sample()
            Sample: Dictionary with correct key that specifies text column and value as text we want to classify
        Returns:
            Dictionary with the input text and the predicted labels
        """

        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = data_module.prepare_sample([sample])
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()

            predictions = (logits >= 0.5).astype(int) #TODO - arg out this default 0.5

            predicted_labels = {label: predictions[label_idx] for label_idx, label in enumerate(
                self.args.hparams['label'])}
            sample["predicted_labels"] = predicted_labels

        return sample

    def predict_prob(self, data_module: LightningDataModule, sample: dict) -> dict:
        """
        Predict function that returns probability

        Args:
            data_module: module with method prepare_sample()
            sample: Dictionary with correct key that specifies text column and value as text we want to classify
        Returns:
            Dictionary with the input text and the predicted softmax label probability
        """

        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = data_module.prepare_sample([sample])
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()
            return logits

    # Probably can kill all of this commented code
    # def evaluate_live(self, data_module: LightningDataModule):
    #     """
    #     The evaluate live method for the base transformer classifier is just a REPL that processes text and passes
    #     it to the trained model.

    #     Args:
    #         data_module (LightningDataModule): module implementing method prepare_sample()
    #     """
    #     print("Live Demo Mode.\nEnter 'q' or 'quit' (without quotes) to exit the program.\nEnter a single text_sample to run classification on.\n")
    #     while True:
    #         user_input = input("> ")
    #         if user_input == "q" or user_input == "quit":
    #             break
    #         sample = {}
    #         sample[data_module] = user_input.strip()
    #         prediction = self.predict_prob(data_module, sample=sample)
    #         print(prediction)

    # def evaluate_file(self, file_path: str, out_path: str = None):
    #     """
    #     Evaluates a file, with one text_sample on each file, and gets the model prediction for each line. 
    #     Sorts the return values with highest (positives) first, and appends to the out_path.

    #     Args:
    #         file_path (str): input file with one text_sample prediction done per line
    #         out_path (str): output path of sorted predictions. if none, prints formatted to stdout
    #     """
    #     with open(file_path) as fp:
    #         results_dict = {}
    #         for _, line in enumerate(tqdm(fp)):
    #             sample = {}
    #             sample[self.data.text_col] = line.strip()
    #             prediction = self.predict_prob(self.data, sample=sample)
    #             results_dict[line.strip()] = prediction
    #     sorted_results_dict = {k: str(v) for k, v in sorted(
    #         results_dict.items(), key=lambda x: x[1])}
    #     with open(out_path, "w") as fp:
    #         json.dump(sorted_results_dict, fp)

    # TODO - verify these validation, test, predict handlers arent needed to be overridden here and then remove them 1 by 1.
    # the ones that do need to be overridden can go in the multilabel base classifier
    def validation_step(self, batch: tuple, batch_nb: int) -> OrderedDict:
        """[summary]

        Args:
            batch (tuple): [description]
            batch_nb (int): [description]

        Returns:
            OrderedDict: [description]
        """
        return self._shared_evaluation_step(batch, batch_nb, stage="val")

    # def validation_step_end(self, outputs: dict):
    #     """Synchronizes metrics across GPU's in DP mode by updating and computing given the dictionary
    #     of outputs from the validation_step called on each GPU

    #     Args:
    #         outputs (dict): Return value of validation_step

    #     """
    #     output_metrics = self.metrics.compute_metrics(outputs['logits'], outputs['target'], stage='val')
    #     self.log_dict(output_metrics, on_step=False, on_epoch=True)
    #     return outputs

    def training_epoch_end(self, outputs: List) -> dict:
        """Runs pytorch lightning validation_epoch_end_function. For more details, see
        _run_epoch_end_metrics

        Args:
            outputs - list of dictionaries returned by step, across multiple gpus.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        cms = self._create_confusion_matrices(outputs)
        for label_name, confusion_matrix in cms.items():
            self.logger.experiment.log({"train/epoch_confusion_matrix/{}".format(label_name): confusion_matrix})
        return None
    
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        """Runs pytorch lightning validation_epoch_end_function. For more details, see
        _run_epoch_end_metrics

        Args:
            outputs - list of dictionaries returned by step, across multiple gpus.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        cms = self._create_confusion_matrices(outputs)
        for label_name, confusion_matrix in cms.items():
            self.logger.experiment.log({"val/epoch_confusion_matrix/{}".format(label_name): confusion_matrix})
        return None

    def test_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        """[summary]

        Args:
            batch (tuple): [description]
            batch_idx (int): [description]

        Returns:
            OrderedDict: [description]
        """
        return self._shared_evaluation_step(batch, batch_idx, stage="test")

    # def test_step_end(self, outputs: List) -> Dict:
    #     """[summary]

    #     Args:
    #         outputs (list): [description]

    #     Returns:
    #         dict: [description]
    #     """

    #     output_metrics = self.metrics.compute_metrics(outputs['logits'], outputs['target'], stage='test')
    #     self.log_dict(output_metrics, on_step=False, on_epoch=True)
    #     return outputs

    def test_epoch_end(self, outputs: List[Dict]):
        """[summary]

        Args:
            outputs (List[Dict]): [description]

        Returns:
            [type]: [description]
        """
        cms = self._create_confusion_matrices(outputs)
        for label_name, confusion_matrix in cms.items():
            self.logger.experiment.log({"test/epoch_confusion_matrix/{}".format(label_name): confusion_matrix})
        return None

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int=0):
        """PyTorch Lightning function to do raw batch prediction

        Args:
            batch (tuple): [description]
            batch_idx (int): [description]
            dataloader_idx (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """
        inputs, _, ids = batch
        model_out = self.forward(**inputs)
        logits = model_out["logits"]

        sample_ids = ids['sample_id_keys']

        output = OrderedDict({
            "logits": logits,
            "sample_id_keys": sample_ids,
        })

        return output
    
    #NOTE - PyTorch Lightning 1.5.1 still uses this on_ prefix for predict_step_end, but this may change soon. see here: https://github.com/PyTorchLightning/pytorch-lightning/issues/9380
    def on_predict_step_end(self, outputs: List[Dict]) -> List:
        return outputs
    
    #NOTE - PyTorch Lightning 1.5.1 still uses this on_ prefix for predict_epoch_end, but this may change soon. see here: https://github.com/PyTorchLightning/pytorch-lightning/issues/9380
    def on_predict_epoch_end(self, outputs: List) -> Dict:
        if self.live_eval_mode:
            prediction_logits = outputs[0][0]["logits"].cpu().squeeze()
            prediction_softmax = torch.nn.Softmax(dim=0)(prediction_logits)
            output_str = "\nPrediction:\n"
            for label_idx, label in enumerate(self.data.label_encoder.vocab):
                output_str += "\t{}:{}\n".format(label, prediction_softmax[label_idx])
            self.logger.info(output_str)
        else:
            self.data._write_predictions(outputs[0])
        return None


    #This probably becomes the shared eval step
    def _shared_evaluation_step(self, batch: tuple, batch_idx: int, stage: str) -> OrderedDict:
        """[summary]

        Args:
            batch (tuple): [description]
            batch_idx (int): [description]
            stage (str): [description]

        Returns:
            OrderedDict: [description]
        """
        inputs, targets, _ = batch
        model_out = self.forward(**inputs)

        # We want to get an overall picture of the loss but also track each label individually.
        # Overall loss can be tracked on the step
        loss_val = self.loss(model_out, targets)
        labels = targets["labels"]
        logits = model_out["logits"]
        preds = (logits >= 0.5).float()  # should use a threshold argument
        val_acc = torch.sum(labels == preds).item() / (len(labels) * 1.0)
        val_acc = torch.tensor(val_acc)

        # if self.on_gpu:
        #     val_acc = val_acc.cuda(loss_val.device.index)

        # # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss_val = loss_val.unsqueeze(0)
        #     val_acc = val_acc.unsqueeze(0)

        loss_key = stage + '/loss'
        acc_key = stage + '/acc'

        output = OrderedDict({
            loss_key: loss_val,
            acc_key: val_acc,
            "logits": logits,
            "pred": preds,
            "target": labels
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output
    
    def _create_confusion_matrices(self, outputs: Dict) -> Dict:
        """[summary]

        Args:
            outputs (Dict): [description]

        Returns:
            Dict: [description]
        """
        cms = {}
        for label_idx, label in enumerate(self.data.label_encoder.vocab):
            logits = torch.cat([output['logits'][:, label_idx]
                            for output in outputs]).detach().cpu().numpy()
            pred = (logits >= 0.5).astype(int) #TODO - arg out this default 0.5
            trg = torch.cat([output['target'][:, label_idx]
                                for output in outputs]).detach().cpu().numpy()

            # predictions = self.data.label_encoder.batch_decode(pred)
            # target = self.data.label_encoder.batch_decode(trg)
            # cm = wandb.plot.confusion_matrix(y_true=target, preds=predictions, class_names=self.data.label_encoder.vocab)
            cm = wandb.plot.confusion_matrix(y_true=trg, preds=pred, class_names=["Not_{}".format(label), label])
            cms[label] = cm
        return cms

    # def _create_confusion_matrix_OLD(self, predictions: torch.tensor, target: torch.tensor, label_name: str):
    #     """
    #     Given predictions and targets tensors, create a visual confusion matrix from matplotlib that can
    #     be loaded into weights and biases under the 'Media' tab. Use this pattern to add more visuals.

    #     Args:
    #         predictios: torch.tensor: the raw numeric predictions of the classifier.
    #         targets: torch.tensor: the raw numeric target labels of the classifier.
    #         label_name: str, the (positive) name for the label
    #     """
    #     predictions_string = [label_name if val == 1 else "not_{}".format(
    #         label_name) for val in predictions.numpy()]
    #     target_string = [label_name if val == 1 else "not_{}".format(
    #         label_name) for val in target.numpy()]

    #     vocab = ["not_{}".format(label_name), label_name]

    #     confmatrix = confusion_matrix(
    #         predictions_string, target_string, labels=vocab)
    #     confdiag = np.eye(len(confmatrix)) * confmatrix
    #     np.fill_diagonal(confmatrix, 0)

    #     confmatrix = confmatrix.astype('float')
    #     n_confused = np.sum(confmatrix)
    #     confmatrix[confmatrix == 0] = np.nan
    #     confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': vocab, 'y': vocab, 'z': confmatrix,
    #                              'hoverongaps': False, 'hovertemplate': 'Predicted %{y}<br>instead of %{x}<br>on %{z} examples<extra></extra>'})

    #     confdiag = confdiag.astype('float')
    #     n_right = np.sum(confdiag)
    #     confdiag[confdiag == 0] = np.nan
    #     confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': vocab, 'y': vocab, 'z': confdiag,
    #                            'hoverongaps': False, 'hovertemplate': 'Predicted %{y} correctly<br>on %{z} examples<extra></extra>'})

    #     fig = go.Figure((confdiag, confmatrix))
    #     transparent = 'rgba(0, 0, 0, 0)'
    #     n_total = n_right + n_confused
    #     fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.05)'], [
    #                       1, f'rgba(180, 0, 0, {max(0.2, (n_confused/n_total) ** 0.5)})']], 'showscale': False}})
    #     fig.update_layout({'coloraxis2': {'colorscale': [[0, transparent], [
    #                       0, f'rgba(0, 180, 0, {min(0.8, (n_right/n_total) ** 2)})'], [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})

    #     xaxis = {'title': {'text': 'y_true'}, 'showticklabels': False}
    #     yaxis = {'title': {'text': 'y_pred'}, 'showticklabels': False}

    #     fig.update_layout(title={'text': 'Confusion Matrix: {}'.format(label_name), 'x': 0.5},
    #                       paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)

    #     return wandb.data_types.Plotly(fig)

    # def validation_epoch_end_OLD(self, outputs: list) -> dict:
    #     """ Function that takes as input a list of dictionaries returned by the validation_step
    #     function and measures the model performance accross the entire validation set.

    #     A lot of this logic here defines what to do in the case of multiple GPU's. In this case,
    #     what we do with the validation loss is average the number out across the GPU's.

    #     For other metrics, we can detach all of the predictions and targets from the GPU, concatenate them,
    #     and just calculate them as one large vector.

    #     For this mulitlabel, we do an extension of what is in the base_classifier for this method.
    #     We report the general standard metrics for everything averaged together, but also for
    #     each of the individual labels. This results in four metrics per label, plus four metrics
    #     for the labels combined so (num_labels+1)*4 tracked metrics, plus the standard ones (loss_val, 
    #     loss_epoch, loss_step, epoch, global_step)

    #     Args:
    #         outputs - list of dictionaries returned by validation step, across multiple gpus.

    #     Returns:
    #         - Dictionary with metrics to be added to the lightning logger.  
    #     """
    #     val_loss_mean = 0
    #     val_acc_mean = 0

    #     pred = torch.cat([output['pred']
    #                       for output in outputs]).detach().cpu()
    #     target = torch.cat([output['target']
    #                         for output in outputs]).detach().cpu()

    #     # The individual labels can be tracked at the end
    #     individual_labels = {}
    #     for label_index, label in enumerate(self.args.hparams['label']):
    #         target_tensor = target[:, label_index]
    #         pred_tensor = pred[:, label_index]
    #         individual_labels[label] = {
    #             "pred": pred_tensor, "target": target_tensor
    #         }

    #     # Set certain metric outputs to 0 if there are no positives.
    #     total_positives = torch.sum(target).cpu().numpy(
    #     ) if self.on_gpu else torch.sum(target).numpy()

    #     if total_positives == 0:
    #         wandb.termwarn(
    #             "Warning, no sample targets in ANY labels were found that are positive. Setting certain metrics to output 0.")
    #         # 1 or 0 is standard. 0 is nice though because then you go up from the beginning :)
    #         auroc = f1 = precision = recall = torch.tensor([0.])
    #     else:
    #         flat_pred = torch.flatten(pred)
    #         flat_target = torch.flatten(target)
    #         f1 = self.metrics['f1'](flat_target.numpy(), flat_pred.numpy(), average='macro')
    #         auroc = self.metrics['auroc'](flat_pred, flat_target)
    #         precision = self.metrics['precision'](flat_target.numpy(), flat_pred.numpy(), average='macro')
    #         recall = self.metrics['recall'](flat_target.numpy(), flat_pred.numpy(), average='macro')

    #         for label_index, label in enumerate(self.args.hparams['label']):
    #             flat_target_tensor = torch.flatten(
    #                 individual_labels[label]["target"])
    #             flat_pred_tensor = torch.flatten(
    #                 individual_labels[label]["pred"])
    #             true_pos_count = torch.sum(flat_target_tensor)
    #             if true_pos_count == 0:
    #                 wandb.termwarn(
    #                     "Warning, no sample targets in label {} were found that are positive. Setting certain metrics to output 0.".format(label))
    #             individual_labels[label]["f1"] = self.metrics['f1'](
    #                 flat_target_tensor.numpy(), flat_pred_tensor.numpy()) if true_pos_count > 0 else torch.tensor([0.])
    #             individual_labels[label]["auroc"] = self.metrics['auroc'](
    #                 flat_pred_tensor, flat_target_tensor) if true_pos_count > 0 else torch.tensor([0.])
    #             individual_labels[label]["precision"] = self.metrics['precision'](
    #                 flat_target_tensor.numpy(), flat_pred_tensor.numpy()) if true_pos_count > 0 else torch.tensor([0.])
    #             individual_labels[label]["recall"] = self.metrics['recall'](
    #                 flat_target_tensor.numpy(), flat_pred_tensor.numpy()) if true_pos_count > 0 else torch.tensor([0.])
    #             individual_labels[label]["cm"] = self._create_confusion_matrix(
    #                 flat_pred_tensor, flat_target_tensor, label)

    #     # We will use the mean loss across all of the GPU's as the loss
    #     for output in outputs:
    #         val_loss = output["val_loss"]

    #         # reduce manually when using dp or ddp2
    #         if self.trainer.use_dp or self.trainer.use_ddp2:
    #             val_loss = torch.mean(val_loss)
    #         val_loss_mean += val_loss

    #         # reduce manually when using dp
    #         val_acc = output["val_acc"]
    #         if self.trainer.use_dp or self.trainer.use_ddp2:
    #             val_acc = torch.mean(val_acc)

    #         val_acc_mean += val_acc

    #     val_loss_mean /= len(outputs)
    #     val_acc_mean /= (len(outputs)*len(self.args.hparams['label']))
    #     tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}

    #     # track the global metrics
    #     tracked_metrics = {
    #         "val_loss": val_loss_mean,
    #         "val_acc": val_acc_mean,
    #         'f1': f1,
    #         'auroc': auroc,
    #         'precision': precision,
    #         'recall': recall
    #     }

    #     # track each individual label metric
    #     for label in self.args.hparams['label']:
    #         tracked_metrics['f1_{}'.format(
    #             label)] = individual_labels[label]["f1"]
    #         tracked_metrics['auroc_{}'.format(
    #             label)] = individual_labels[label]["auroc"]
    #         tracked_metrics['precision_{}'.format(
    #             label)] = individual_labels[label]["precision"]
    #         tracked_metrics['recall_{}'.format(
    #             label)] = individual_labels[label]["recall"]
    #         tracked_metrics['cm_{}'.format(
    #             label)] = individual_labels[label]["cm"]

    #     self.log_metrics(tracked_metrics)
    #     self.log("recall", recall, prog_bar=True)
    #     self.log("precision", precision, prog_bar=True)
    #     self.log("f1", f1, prog_bar=True)def validation_epoch_end_OLD(self, outputs: list) -> dict:
    #     """ Function that takes as input a list of dictionaries returned by the validation_step
    #     function and measures the model performance accross the entire validation set.

    #     A lot of this logic here defines what to do in the case of multiple GPU's. In this case,
    #     what we do with the validation loss is average the number out across the GPU's.

    #     For other metrics, we can detach all of the predictions and targets from the GPU, concatenate them,
    #     and just calculate them as one large vector.

    #     For this mulitlabel, we do an extension of what is in the base_classifier for this method.
    #     We report the general standard metrics for everything averaged together, but also for
    #     each of the individual labels. This results in four metrics per label, plus four metrics
    #     for the labels combined so (num_labels+1)*4 tracked metrics, plus the standard ones (loss_val, 
    #     loss_epoch, loss_step, epoch, global_step)

    #     Args:
    #         outputs - list of dictionaries returned by validation step, across multiple gpus.

    #     Returns:
    #         - Dictionary with metrics to be added to the lightning logger.  
    #     """
    #     val_loss_mean = 0
    #     val_acc_mean = 0

    #     pred = torch.cat([output['pred']
    #                       for output in outputs]).detach().cpu()
    #     target = torch.cat([output['target']
    #                         for output in outputs]).detach().cpu()

    #     # The individual labels can be tracked at the end
    #     individual_labels = {}
    #     for label_index, label in enumerate(self.args.hparams['label']):
    #         target_tensor = target[:, label_index]
    #         pred_tensor = pred[:, label_index]
    #         individual_labels[label] = {
    #             "pred": pred_tensor, "target": target_tensor
    #         }

    #     # Set certain metric outputs to 0 if there are no positives.
    #     total_positives = torch.sum(target).cpu().numpy(
    #     ) if self.on_gpu else torch.sum(target).numpy()

    #     if total_positives == 0:
    #         wandb.termwarn(
    #             "Warning, no sample targets in ANY labels were found that are positive. Setting certain metrics to output 0.")
    #         # 1 or 0 is standard. 0 is nice though because then you go up from the beginning :)
    #         auroc = f1 = precision = recall = torch.tensor([0.])
    #     else:
    #         flat_pred = torch.flatten(pred)
    #         flat_target = torch.flatten(target)
    #         f1 = self.metrics['f1'](flat_target.numpy(), flat_pred.numpy(), average='macro')
    #         auroc = self.metrics['auroc'](flat_pred, flat_target)
    #         precision = self.metrics['precision'](flat_target.numpy(), flat_pred.numpy(), average='macro')
    #         recall = self.metrics['recall'](flat_target.numpy(), flat_pred.numpy(), average='macro')

    #         for label_index, label in enumerate(self.args.hparams['label']):
    #             flat_target_tensor = torch.flatten(
    #                 individual_labels[label]["target"])
    #             flat_pred_tensor = torch.flatten(
    #                 individual_labels[label]["pred"])
    #             true_pos_count = torch.sum(flat_target_tensor)
    #             if true_pos_count == 0:
    #                 wandb.termwarn(
    #                     "Warning, no sample targets in label {} were found that are positive. Setting certain metrics to output 0.".format(label))
    #             individual_labels[label]["f1"] = self.metrics['f1'](
    #                 flat_target_tensor.numpy(), flat_pred_tensor.numpy()) if true_pos_count > 0 else torch.tensor([0.])
    #             individual_labels[label]["auroc"] = self.metrics['auroc'](
    #                 flat_pred_tensor, flat_target_tensor) if true_pos_count > 0 else torch.tensor([0.])
    #             individual_labels[label]["precision"] = self.metrics['precision'](
    #                 flat_target_tensor.numpy(), flat_pred_tensor.numpy()) if true_pos_count > 0 else torch.tensor([0.])
    #             individual_labels[label]["recall"] = self.metrics['recall'](
    #                 flat_target_tensor.numpy(), flat_pred_tensor.numpy()) if true_pos_count > 0 else torch.tensor([0.])
    #             individual_labels[label]["cm"] = self._create_confusion_matrix(
    #                 flat_pred_tensor, flat_target_tensor, label)

    #     # We will use the mean loss across all of the GPU's as the loss
    #     for output in outputs:
    #         val_loss = output["val_loss"]

    #         # reduce manually when using dp or ddp2
    #         if self.trainer.use_dp or self.trainer.use_ddp2:
    #             val_loss = torch.mean(val_loss)
    #         val_loss_mean += val_loss

    #         # reduce manually when using dp
    #         val_acc = output["val_acc"]
    #         if self.trainer.use_dp or self.trainer.use_ddp2:
    #             val_acc = torch.mean(val_acc)

    #         val_acc_mean += val_acc

    #     val_loss_mean /= len(outputs)
    #     val_acc_mean /= (len(outputs)*len(self.args.hparams['label']))
    #     tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}

    #     # track the global metrics
    #     tracked_metrics = {
    #         "val_loss": val_loss_mean,
    #         "val_acc": val_acc_mean,
    #         'f1': f1,
    #         'auroc': auroc,
    #         'precision': precision,
    #         'recall': recall
    #     }

    #     # track each individual label metric
    #     for label in self.args.hparams['label']:
    #         tracked_metrics['f1_{}'.format(
    #             label)] = individual_labels[label]["f1"]
    #         tracked_metrics['auroc_{}'.format(
    #             label)] = individual_labels[label]["auroc"]
    #         tracked_metrics['precision_{}'.format(
    #             label)] = individual_labels[label]["precision"]
    #         tracked_metrics['recall_{}'.format(
    #             label)] = individual_labels[label]["recall"]
    #         tracked_metrics['cm_{}'.format(
    #             label)] = individual_labels[label]["cm"]

    #     self.log_metrics(tracked_metrics)
    #     self.log("recall", recall, prog_bar=True)
    #     self.log("precision", precision, prog_bar=True)
    #     self.log("f1", f1, prog_bar=True)
    #     self.log("auroc", auroc, prog_bar=True)
    #     self.log("val_loss", val_loss_mean, prog_bar=True)
    #     self.log("val_acc", val_acc_mean, prog_bar=True)
    #     return None
    #     self.log("auroc", auroc, prog_bar=True)
    #     self.log("val_loss", val_loss_mean, prog_bar=True)
    #     self.log("val_acc", val_acc_mean, prog_bar=True)
    #     return None

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        return parser