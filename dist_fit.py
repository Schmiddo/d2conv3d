from trainer import Trainer

def fit(cfg, model, train_data, val_data, test_data):
  trainer = Trainer(cfg)
  trainer.fit(model, train_data, val_data, test_data)

