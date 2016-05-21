package org.deeplearning4j.examples.convolution;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//import org.deeplearning4j.nn.conf.LearningRatePolicy;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Lakini on 20/05/16.
 */
public class ConvolutionNetLFW {
    private static final Logger log = LoggerFactory.getLogger(ConvolutionNetLFW.class);

    public static void main(String[] args) throws Exception {

        //depth of the images is 1
        int nChannels = 1;
        //Number of classes in LFW is 5749
        int outputNum = 5749;
        int nEpochs = 10;
        int iterations = 1;
        int seed = 123;

        //Path of the labeled images
        String labeledPath = "C:/Users/Lakini/Documents/datasets-DL4J/lfw/lfw/lfw";

        //create array of strings called labels
        List<String> labels = new ArrayList<>();

        //traverse dataset to get each label
        for(File f : new File(labeledPath).listFiles()){
            labels.add(f.getName());
        }

        // Instantiating RecordReader. LFW photos has 28 x 28 photos
        RecordReader recordReader = new ImageRecordReader(28, 28, true, labels);
        recordReader.initialize(new FileSplit(new File(labeledPath)));

        log.info("Load data....");
        // Convert Canova to DL4J
        org.deeplearning4j.datasets.iterator.DataSetIterator iter = new RecordReaderDataSetIterator(recordReader,784, labels.size());
        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();

        log.info("Split data....");
        //Split data in to training and test data
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(0.75);
        DataSet lfwtrain = testAndTrain.getTrain();
        DataSet lfwtest = testAndTrain.getTest();

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,28,28,1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(lfwtrain);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(lfwtest.iterator().hasNext()){
                DataSet ds = lfwtest;
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }
            log.info(eval.stats());
        }
        log.info("****************Example finished********************");
    }
}
