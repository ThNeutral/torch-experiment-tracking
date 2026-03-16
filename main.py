import torchvision
from utils import create_image_dataloaders


def main():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

    transforms = weights.transforms()
    print(f"Transforms: {transforms}")

    train_dataloader, test_dataloader, class_names = create_image_dataloaders(
        dir="data/pizza_steak_sushi",
        train_transform=transforms,
        test_transform=transforms,
        batch_size=32,
        num_workers=1
    )

    print(f"Class names: {class_names}")


if __name__ == "__main__":
    main()
