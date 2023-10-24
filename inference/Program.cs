using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using CommandLine;

public static class Program
{

    private class Options
    {
        [Option('g', "gpu", Required = false, HelpText = "enable gpu usage.")]
        public bool Gpu { get; set; }

        [Option('m', "model", Required = true, HelpText = "model to use.")]
        public string? Model { get; set; }

        [Option('i', "image", Required = true, HelpText = "image to classify.")]
        public string? Image { get; set; }

        [Option('v', "verbose", Required = false, HelpText = "enable verbose logging.")]
        public bool Verbose { get; set; }

        [Option('c', "classes", Required = true, HelpText = "classes file to be used to translate inferences.")]
        public string? Classes { get; set; }
    }

    const string INPUT_NAME = "input";

    static void ExitWithModelError()
    {
        Console.WriteLine($"- model should have only one input named '{INPUT_NAME}' with four float dimensions.");
        Environment.Exit(2);
    }

    static void ExitWithFileNotFound(string filename)
    {
        Console.WriteLine($"- file {filename} not found.");
        Environment.Exit(2);
    }

    private static string[] LoadClasses(string filename)
    {
        if (!File.Exists(filename))
        {
            ExitWithFileNotFound(filename);
        }

        return File.ReadAllLines(filename);
    }

    private static List<NamedOnnxValue> LoadAndConvertImage(string filename, int width, int height)
    {
        if (!File.Exists(filename))
        {
            ExitWithFileNotFound(filename);
        }

        using Image<Rgb24> image = Image.Load<Rgb24>(filename);

        image.Mutate(x =>
                   {
                       x.Resize(new ResizeOptions
                       {
                           Size = new Size(width, height),
                           Mode = ResizeMode.Crop
                       });
                   });

        Tensor<float> input = new DenseTensor<float>(new[] { 1, height, width, 3 });
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    input[0, y, x, 0] = pixelSpan[x].R;
                    input[0, y, x, 1] = pixelSpan[x].G;
                    input[0, y, x, 2] = pixelSpan[x].B;

                }
            }
        });

        return new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor<float>(INPUT_NAME, input)
            };
    }

    private static (InferenceSession session, int width, int height) CreateONNXRuntime(string filename, bool useGPU, bool verbose)
    {
        if (!File.Exists(filename))
        {
            ExitWithFileNotFound(filename);
        }

        var options = new SessionOptions();

        if (useGPU)
        {
            var cuda = new OrtCUDAProviderOptions();
            options = SessionOptions.MakeSessionOptionWithCudaProvider(cuda);
        }

        if (verbose)
        {
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
            options.LogVerbosityLevel = 5;
        }

        var session = new InferenceSession(filename, options);

        if (session.InputMetadata.Count != 1)
        {
            ExitWithModelError();
        }

        var inputName = session.InputMetadata.First().Key;
        var imageInput = session.InputMetadata.First().Value;

        if ((inputName != INPUT_NAME) || (imageInput.Dimensions.Length != 4) || (imageInput.ElementDataType != TensorElementType.Float))
        {
            ExitWithModelError();
        }

        return (session, imageInput.Dimensions[2], imageInput.Dimensions[1]);
    }

    public static int Main(string[] args)
    {
        var opts = Parser.Default.ParseArguments<Options>(args);

        // If all command line parameters are ok, then proceed.
        if (opts.Errors.Count() == 0)
        {
            var classes = LoadClasses(opts.Value.Classes);

            // Load the model, and load and convert the image for the model size..
            var (session, width, height) = CreateONNXRuntime(opts.Value.Model, opts.Value.Gpu, opts.Value.Verbose);
            var image = LoadAndConvertImage(opts.Value.Image, width, height);

            // Classify the image.
            using var outputs = session.Run(image);

            // What class is the one with the highest probability?
            var output = ((DenseTensor<float>)outputs.Single().Value).ToArray();

            var predictedProbability = output.Max();
            int predictedClass = output.ToList().IndexOf(predictedProbability);

            Console.WriteLine($"The predicted class is {classes[predictedClass]} with probability {predictedProbability}.");

            return 0;
        }

        return 2;
    }
}
