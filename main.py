from models import TinyVGG

def main():
    model_0 = TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=3
    )


if __name__ == "__main__":
    main()
