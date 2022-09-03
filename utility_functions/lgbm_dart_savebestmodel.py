class SaveModelCallback:
    def __init__(self,
                 models_folder: pathlib.Path,
                 fold_id: int,
                 min_score_to_save: float,
                 every_k: int,
                 order: int = 0):
        self.min_score_to_save: float = min_score_to_save
        self.every_k: int = every_k
        self.current_score = min_score_to_save
        self.order: int = order
        self.models_folder: pathlib.Path = models_folder
        self.fold_id: int = fold_id

    def __call__(self, env):
        iteration = env.iteration
        score = env.evaluation_result_list[3][2]
        if iteration % self.every_k == 0:
            print(f'iteration {iteration}, score={score:.5f}')
            if score > self.current_score:
                self.current_score = score
                for fname in self.models_folder.glob(f'original_fold{self.fold_id}_seed42*'):
                    fname.unlink()
                print(f'High Score: iteration {iteration}, score={score:.5f}')
                joblib.dump(env.model, self.models_folder / f'original_fold{self.fold_id}_seed42_{score:.5f}.pkl')


def save_model(models_folder: pathlib.Path, fold_id: int, min_score_to_save: float = 0.79, every_k: int = 100):
    return SaveModelCallback(models_folder=models_folder, fold_id=fold_id, min_score_to_save=min_score_to_save, every_k=every_k)
