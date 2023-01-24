import torch
import numpy as np

from torch.utils.data import DataLoader
from . import LOSS_NAMES, MODEL_NAMES, EVAL_NAMES
from .utils import parse_cfg_str


def get_accuracy(test_data: DataLoader, model, class_sizes: [int],
                 calc_avg=True, calc_perclass=True, top=1, device: torch.device = torch.device("cpu")):
    """Return a tuple containing the accuracy values of a given model.

    The first element of the returned tuple is the average accuracy of the model.
    The second element is a tensor containing the accuracy of each class
      separately.

    The parameters calc_avg and calc_perclass exist for performance adjustments.
    If calc_avg is False, the average accuracy isn't calculated and left as 0.
    If calc_perclass is False, the per-class accuracies aren't calculated and left
      as 0.
    """
    
    total_size = 0
    
    with torch.no_grad():
        num_labels = len(class_sizes)
        
        per_class_acc = torch.zeros(num_labels, dtype=torch.float32, device=device)
        avg_acc = float(0)
        
        for num_batch, (inp, target) in enumerate(test_data):
            inp = inp.double().to(device)
            target = target.to(device)
            output = model(inp).to(device)
            
            if top == 1:
                result = (torch.argmax(output, dim=1) == target)
            else:
                result = []
                top_preds = np.argpartition(output, -top)[:, -top:]
                
                for i, t in enumerate(target):
                    result.append(t.item() in top_preds[i])
                
                result = torch.Tensor(result)
            
            batch_len = result.shape[0]
            
            if calc_avg:
                avg_acc += result.sum().item()
            
            if calc_perclass:
                for i in range(batch_len):
                    per_class_acc[target[i]] += result[i]
            
            total_size += batch_len
        
        # Average accuracy of the whole test dataset
        if calc_avg:
            avg_acc /= total_size
        
        if calc_perclass:
            for i in range(num_labels):
                # Average accuracy of every class separately
                per_class_acc[i] /= class_sizes[i]
        
        return avg_acc, per_class_acc.tolist()


def evaluate(cfg, train_results, test_dl, test_class_sizes, device: torch.device = torch.device("cpu")):
    eval_list = cfg.Evaluation
    
    for e in eval_list:
        method_name = e.method_name
        method_params = e.method_params

        calc_avg = method_params.calc_avg
        calc_perclass = method_params.calc_perclass
        top = parse_cfg_str(method_params.top, int)
        
        if not (calc_avg or calc_perclass):
            print("Both average and per-class accuracy calculation options were disabled, returning.")
            return
        
        print("Starting evaluation with method:", EVAL_NAMES[method_name])
        
        if method_name == "get_accuracy":
            for r in train_results:
                avg_acc, perclass_acc = get_accuracy(test_dl, r["model"], test_class_sizes, calc_avg=calc_avg,
                                                     calc_perclass=calc_perclass, top=top, device=device)
                
                print(
                    MODEL_NAMES[r["model_name"]]
                    + " trained with "
                    + LOSS_NAMES[r["loss_name"]]
                    + (" using training parameters " + str(r["options"]) if method_params.print_task_options else "")
                    + ":"
                )

                if calc_avg:
                    print(f"  Average top-{top} accuracy:", str(avg_acc) + "%")
                if calc_perclass:
                    print(f"Top-{top} accuracy per class:", perclass_acc)
                
                print()  # Empty line
        else:
            raise Exception("Invalid method name encountered during evaluation: " + method_name)

        print()  # Print empty line
