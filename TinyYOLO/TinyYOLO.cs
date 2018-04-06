using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Windows.Media;
using Windows.Storage;
using Windows.AI.MachineLearning.Preview;

// TinyYOLO

namespace TinyYOLO
{
    public sealed class TinyYOLOModelInput
    {
        public VideoFrame image { get; set; }
    }

    public sealed class TinyYOLOModelOutput
    {
        public IList<float> grid { get; set; }
        public TinyYOLOModelOutput()
        {
            this.grid = new List<float>();
        }
    }

    public sealed class TinyYOLOModel
    {
        private LearningModelPreview learningModel;
        public static async Task<TinyYOLOModel> CreateTinyYOLOModel(StorageFile file)
        {
            LearningModelPreview learningModel = await LearningModelPreview.LoadModelFromStorageFileAsync(file);
            TinyYOLOModel model = new TinyYOLOModel();
            model.learningModel = learningModel;
            return model;
        }
        public async Task<TinyYOLOModelOutput> EvaluateAsync(TinyYOLOModelInput input) {
            TinyYOLOModelOutput output = new TinyYOLOModelOutput();
            LearningModelBindingPreview binding = new LearningModelBindingPreview(learningModel);
            binding.Bind("image", input.image);
            binding.Bind("grid", output.grid);
            LearningModelEvaluationResultPreview evalResult = await learningModel.EvaluateAsync(binding, string.Empty);
            return output;
        }
    }
}
