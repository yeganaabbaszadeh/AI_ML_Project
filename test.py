import torch
from torchmetrics import F1Score
from models.model import ResNet18, VGG16

from torch.utils.data import DataLoader
from datasets.dataset_retrieval import custom_dataset
import tqdm


def test(model, data_test):
    f1score = 0
    f1 = F1Score(num_classes=107, task='multiclass')
    data_iterator = enumerate(data_test)  # take batches
    f1_list = []
    f1t_list = []

    with torch.no_grad():
        model.eval()  # switch model to evaluation mode
        tq = tqdm.tqdm(total=len(data_test))
        tq.set_description('test:')

        total_loss = 0

        for _, batch in data_iterator:
            # forward propagation
            image, label = batch
            image = image.cuda()
            label = label.cuda()
            pred = model(image)

            pred = pred.softmax(dim=1)

            f1_list.extend(torch.argmax(pred, dim=1).tolist())
            f1t_list.extend(torch.argmax(label, dim=1).tolist())
            #f1score += f1_score(label.squeeze().detach().cpu(), pred.squeeze().detach().cpu())

            #total_loss += loss.item()
            tq.update(1)

    tq.close()
    print("F1 score: ", f1(torch.tensor(f1_list), torch.tensor(f1t_list)))

    return None


test_data = custom_dataset("test")


test_loader = DataLoader(
    test_data,
    batch_size=8
)

model = ResNet18(107).cuda()
# model = VGG16(107).cuda()

checkpoint = torch.load("checkpoints/vgg16sgd.pth")

model.load_state_dict(checkpoint['state_dict'])

test(model, test_loader)
