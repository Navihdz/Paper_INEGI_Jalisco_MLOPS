
import hydra

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def examplefunction(cfg):
    print(cfg)

    for params in cfg.grid_search:
        learning_rate = float(params.learning_rate)
        batch_size = int(params.batch_size)
        num_epochs = int(params.num_epochs)

        """This is an example function"""
        print("Hello World", learning_rate)
        print("Hello World2", batch_size)
        print("Hello World3", num_epochs)

if __name__ == "__main__":
    examplefunction()
