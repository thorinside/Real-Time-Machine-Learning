// test_wolf.cpp -- main program to test Learning Feature Trees using WOLF approximation for machine learning
/*
Copyright (c) 2024 William Ward Armstrong
MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. */

#include <iostream>
#include <fstream>
#include <chrono>
#include <assert.h>
#include "mnist_file.h"
#include "wolf.h"
const char* train_image_path = "..\\Data\\train-images.idx3-ubyte"; // Original MNIST images
const char* train_label_path = "..\\Data\\train-labels.idx1-ubyte"; // Original MNIST labels
mnist_image_t* train_images;
uint8_t* train_labels;
size_t treeNo; // Index of an LF_tree
extern long leafCount;
extern bool treeFinal; // an SB does not change after it is made final, treeFinal means all SBs in the tree are final
class cLF_tree;
uint32_t numberofimages = 60000;
long nSensors;
int numberLF_trees;
int minSamples2Split;
int constancyLimit;
double convolutionRadius{};
void loadMNISTdata();
double reinforcement(uint8_t action, uint8_t label);
void testOnTRAINING_SET(int numberLF_trees, cLF_tree** apLF_tree);
void testOnTEST_SET(int numberLF_trees, cLF_tree** apLF_tree);
void view(mnist_image_t* pimage, uint8_t label); // Shows crude copies of the badly classified MNIST test images

int main(int argc, char* argv[])
{
    std::cout << "Learning Feature Trees using WOLF approximation for machine learning\n\nInput parameters:\n" << std::endl;
    std::cout << "\nProject name " << argv[1] << "\nNumber of pixels in retina                        " << argv[2] <<
        "\nNumber of Learning Feature Trees (multiple of 10) " << argv[3] << "\nMinimum images per block allowing split           " <<
        argv[4] << "\nSensor constant if std. dev. less than            " << argv[5] <<
        "\nConvolution kernel radius                         " << argv[6] << std::endl << std::endl;
    int argCount = argc;
    if (argc != 7) // We expect arguments as listed here (argc is not counted among them)
    {
        std::cout << " Wrong number of input values! " << std::endl;
        std::cout << "Usage: " << "Data_file_name nSensors numberLF_trees minSamples2Split constandyLimit convolutionRadius" << std::endl;
    }
    char* ProjectName = argv[1];
    nSensors = atoi(argv[2]);
    numberLF_trees = atoi(argv[3]); // We grow a number of trees, some for different tasks, some to increase accuracy for a task.
    minSamples2Split = atoi(argv[4]); // A block with fewer samples can't split
    constancyLimit = atoi(argv[5]); // The closer this is to zero, the fewer variables are removed as "deemed constant"
    convolutionRadius= atof(argv[6]); // This seemed to improve generalization
    loadMNISTdata();
    createKernel(); // This is for the convolution operation on *features*, not images
    // Creating LF_trees  ***************************************************************************
    initCentroidSampleNumbers(nSensors); //sets up two vectors: initC and image_numbers_setup that initialize trees
    // Using several trees improves the statistical average when the trees are shifted copies of the original tree or at least different
    std::cout << "Creating array of " << numberLF_trees << " Learning Feature Trees.\n";
    auto apLF_tree = (cLF_tree**)malloc(numberLF_trees * sizeof(cLF_tree*));
    auto start_first_tree = std::chrono::steady_clock::now();
    double tparallel = 0; // This is the maximum time to grow a single Learning Feature Tree during program execution.
    // In a parallel system, this would be the time growing all trees. For example if growing 200 LF-trees takes 40 minutes
    // to learn to classify the 60000 MNIST numerals, growing the trees in parallel would take only 12seconds, and much less if the parallel code were optimized.
    for (treeNo = 0; treeNo < numberLF_trees; ++treeNo)
    {
        if (treeNo < numberLF_trees)
        {
            apLF_tree[treeNo] = create_LF_tree();
            apLF_tree[treeNo]->setTreeNumber(treeNo);
            apLF_tree[treeNo]->setSensorNumber(nSensors);
            int SBcount = 0; // SB stands for Sample Block. It is a block in the partition created by the hyperplanes of a Learning Feature Tree.
            // Grow an LF_tree  **************************************
           // std::cout << "\nStarting growth of Learning Feature Tree number " << treeNo << std::endl;
            const auto start_tree = std::chrono::steady_clock::now();
            treeFinal = false;
            apLF_tree[treeNo]->loadSBs(60000); // loads all 60000 image numbers into SB number 0 of the new tree
            apLF_tree[treeNo]->growTree();
            //apLF_tree[treeNo]->checkFinalTree(); // Optional -- this provides a rough idea of perfomance on the training data.
            const auto finish_tree = std::chrono::steady_clock::now();
            const auto elapsed0 = std::chrono::duration_cast<std::chrono::seconds>(finish_tree - start_tree);
            if (elapsed0.count() > tparallel) tparallel = (double) elapsed0.count();
            auto elapsed1 = std::chrono::duration_cast<std::chrono::seconds>(finish_tree - start_first_tree);
            std::cout << "Growing tree " << treeNo << " took "  << elapsed0.count() << " sec. Elapsed " << elapsed1.count() <<
                " sec. To go (est.) " << ceil(elapsed1.count() * (numberLF_trees - treeNo - 1)/((treeNo +1) * 60.0)) << " min." << std::endl;
        }
      } // End of for loop growing a number of LF_trees
    auto end_last_tree = std::chrono::steady_clock::now();
    auto trainTime = std::chrono::duration_cast<std::chrono::seconds>(end_last_tree - start_first_tree);
    std::cout << "\n\nProject name " << ProjectName << "\nNumber of pixels in retina                        " << argv[2] << 
        "\nNumber of Learning Feature Trees (multiple of 10) " << argv[3] << "\nMinimum images per block allowing split           " <<
        argv[4] << "\nSensor constant if std. dev. less than            " << argv[5] <<
        "\nConvolution kernel radius                         " << argv[6] << std::endl;
    std::cout << "\nResults of testing on TRAINING DATA --  perhaps useful for develoment \n";
    std::cout << "Mean time to grow a Learning Feature Tree         " << trainTime.count() / (double) numberLF_trees << " seconds." << std::endl;
    std::cout << "Mean leaf count of the Learning Feature Trees     " << leafCount / (double) numberLF_trees << std::endl;
    std::cout << "Estimated parallel training time                  " << tparallel << " seconds. " << std::endl;
    testOnTRAINING_SET(numberLF_trees, apLF_tree);
    free(train_images);
    free(train_labels);
    std::cout << "\n\nResults on TEST DATA (the one that counts!)\n";
    testOnTEST_SET(numberLF_trees, apLF_tree);
    free(apLF_tree);
}  // End of main routine and the program run

double reinforcement(uint8_t action, uint8_t label)  // This training is done by the trainer who knows only the system's action and the label
{
       return (action == label) ? 1 : 0; // What digit is it?  95.95% correct
    // return (label == (uint8_t) (treeNo % 10) ) ? 1 : 0; // The action is always treeNo % 10. What digit is it? 95.95% correct. 
    // return (((label == 3 || label == 8) && action == 1) || ((label != 3 && label != 8) && action != 1)) ? 1 : 0; // Answer 1 or 0:Is it an 8 or a 3? 98.3% correct
    // return (((label == 4 || label == 9) && action == 1) || ((label != 4 && label != 9) && action != 1)) ? 1 : 0; // Answer 1 or 0: Is it a 4 or a 9? 98.86% correct
    // return  (((label == 2 || label == 3 || label == 5 || label == 7 || label == 8 || label == 9) && action == 1) || !(label == 2 || label == 3 || label == 5 || label == 7 || label == 8 || label == 9) && action != 1); // Indicate..
    // by a 1 that there is a sort of bar across the top. Any other output means there isn't a bar. 97.9% correct
    // return ((int) label < 5 && action == label)||((int) label > 4)&& ((int)action > 4) ? 1 : 0; // Concentrate on numerals 0 to 4: What digit is it? 96.04%
    // return((label == 5 && action == label) || (label != 5 && action != 5)) ? 1 : 0; //Just do 5's : Is it a 5? 98.76% correct
    // return  (((label == 2 || label == 3 || label == 5 || label == 7 || label == 8 || label == 9 || label == 0) && action == 1) || !(label == 2 || label == 3 || label == 5 || label == 7 || label == 8 || label == 9 || label == 0) && action != 1); 
    // We believe that using several of tests to analyze the image and putting the information together with the probabilities above, one can improve the classification.
    // N.B. The following reinforcement is "illegal" because the teacher should only know the action and the label, not the treeNo.
    // return((label == (uint8_t)(treeNo % 10) && action == label) || (label != (uint8_t)(treeNo % 10) && action != (uint8_t)(treeNo % 10))) ? 1 : 0; // This is cheating! 99.32 % correct
    // The above incorrectly assumes the teacher knows the treeNo % 10, so it is not reinforcement learning. The part after the || does not say the action decided the label correctly!!
}

void loadMNISTdata()
{
    uint32_t numberofimages;
    train_images = get_images(train_image_path, &numberofimages);
    //std::cout << "Number of training images input " << numberofimages << std::endl;
    train_labels = get_labels(train_label_path, &numberofimages);
    //std::cout << "Number of labels input " << numberofimages << std::endl;
}  // End of loadMNISTdata

void testOnTEST_SET(int numberLF_trees, cLF_tree** apLF_tree)
{
    // Evaluation on the test set using the weighted average of linear functions in leaf SBs of several LF_trees.
    // The tree with number treeNo is associated with the action treeNo % 10, e.g.
    // treeNo = 18 predicts the probability that deciding the image is an 8 will be correct.
    // Using several trees in a weighted average increases the accuracy.
    // Comparing groups of trees with different actions allows the system
    // to choose the action with highest probability of being correct. 
    uint32_t numberoftestimages = 10000;
    const char* test_image_path = "..\\Data\\t10k-images.idx3-ubyte";
    const char* test_labels_path = "..\\Data\\t10k-labels.idx1-ubyte";
    auto test_images = (mnist_image_t*)malloc(numberoftestimages * sizeof(mnist_image_t));
    auto test_labels = (uint8_t*)malloc(numberoftestimages * sizeof(uint8_t));
    test_images = get_images(test_image_path, &numberoftestimages);
   // std::cout << "Number of test images input " << numberoftestimages << std::endl;
    test_labels = get_labels(test_labels_path, &numberoftestimages);
   // std::cout << "Number of labels input" << numberoftestimages << std::endl;
    mnist_image_t* pcurrentImage = nullptr;
    uint8_t label;
    uint8_t chosenAction = 0;
    double maxP_right = 0;
    int lowSumWeightsCount = 0;
    std::vector<double> P_right;
    std::vector<size_t> badRecognitions{};
    P_right.assign(10, 0);
    long goodDecisions = 0;
    auto start_test_set = std::chrono::steady_clock::now();
    for (uint32_t imageNo = 0; imageNo < numberoftestimages; ++imageNo)
    {
        pcurrentImage = test_images + imageNo;
        label = test_labels[imageNo];
        double P_rightTree = 0; // The probability thata given tree would get it right
        double P_rightAction = 0;
        double accuP_right = 0;
        double accu_weights = 0;
        double wt = 0;
        int count = 0;
        // We determine for each action the probability of being right and the weight
        maxP_right = 0.0;
        for(uint8_t action = 0; action < 10; ++action)
        {
            count = 0;
            for (int treeNo = 0; treeNo < numberLF_trees; ++treeNo)
            {
                if (treeNo % 10 == action)
                {
                   P_rightTree = apLF_tree[treeNo]->evalBoundedWeightedSB(pcurrentImage, wt);
                    accuP_right += P_rightTree;
                    accu_weights += wt;
                    ++count;
                }
            }
            // P_rightAction is the probability of being right on the image for the current
            // action using the weighted average of probabilities over several trees.
            P_rightAction = accuP_right / accu_weights;
            if (P_rightAction > maxP_right) // maximize over all actions
            {
                maxP_right = P_rightAction;
                chosenAction = action;
            }
        }
        if (accu_weights <= 0.0001f)
        {
            ++lowSumWeightsCount; // Skip this sample in the output
        }
        else
        {
            if (reinforcement(chosenAction, label) == 1)   // Has the system learned the behaviour that was positively reinforced?
            {
                ++goodDecisions;
            }
            else
            {
                badRecognitions.push_back(imageNo);
            }
        }
    } // Loop  over test images
    auto test_finished = std::chrono::steady_clock::now();
    double ProbabilityOfCorrect = (double)goodDecisions / 10000.0;
    if(lowSumWeightsCount > 0) std::cout << "Low sum weights count " << lowSumWeightsCount << "." << std::endl;
    auto elapsed_test = std::chrono::duration_cast<std::chrono::milliseconds>(test_finished - start_test_set);
    std::cout << "Mean time required to classify an image           " << elapsed_test.count()/10000.0 << " milliseconds. " << std::endl;
    std::cout << "Probability of a correct decision                 " << ProbabilityOfCorrect * 100.0 << " percent " << std::endl;
    char dummy;
    for (size_t i = 0; i < badRecognitions.size(); ++i)
    {
        pcurrentImage = test_images + badRecognitions[i];
        view(pcurrentImage, test_labels[badRecognitions[i]]);
        std::cin >> dummy;
        if(dummy == 's') exit(0);
    }
    
    free(test_images);
    free(test_labels);
} // End of testOnTEST_SET

void testOnTRAINING_SET(int numberLF_trees, cLF_tree** apLF_tree)
{
    // Evaluation on the training set using the weighted average of linear functions in leaf SBs of several LF_trees.
    // The tree with number treeNo is associated with the action treeNo % 10, e.g.
    // treeNo = 18 predicts the probability that deciding the image is an 8 will be correct.
    // Using several trees in a weighted average increases the accuracy.
    // Comparing groups of trees with different actions allows the system
    // to choose the action with highest probability of being correct. 
    uint32_t numberoftestimages = 60000;
    mnist_image_t* pcurrentImage = nullptr;
    uint8_t label;
    uint8_t chosenAction = 0;
    double maxP_right = 0;
    int lowSumWeightsCount = 0;
    std::vector<double> P_right;
    std::vector<int> confusion;
    confusion.assign(100, 0);
    int countConfusion = 0;
    P_right.assign(10, 0);
    long goodDecisions = 0;
    auto start_test_set = std::chrono::steady_clock::now();
    for (uint32_t imageNo = 0; imageNo < numberoftestimages; ++imageNo)
    {
        pcurrentImage = train_images + imageNo;
        label = train_labels[imageNo];
        double P_rightTree = 0; // The probability thata given tree would get it right
        double P_rightAction = 0;
        double accuP_right = 0;
        double accu_weights = 0;
        double wt = 0;
        int count = 0;
        // We determine for each action the probability of being right and the weight
        maxP_right = 0.0;
        for (uint8_t action = 0; action < 10; ++action)
        {
            count = 0;
            for (int treeNo = 0; treeNo < numberLF_trees; ++treeNo)
            {
                if (treeNo % 10 == action)
                {
                    P_rightTree = apLF_tree[treeNo]->evalBoundedWeightedSB(pcurrentImage, wt);
                    accuP_right += P_rightTree;
                    accu_weights += wt;
                    ++count;
                }
            }
            // P_rightAction is the probability of being right on the image for the current
            // action using the weighted average of probabilities over several trees.
            P_rightAction = accuP_right / accu_weights;
            if (P_rightAction > maxP_right) // maximize over all actions
            {
                maxP_right = P_rightAction;
                chosenAction = action;
            }
        }
        if (accu_weights <= 0.0001f)
        {
            ++lowSumWeightsCount; // Skip this sample in the output
        }
        else
        {
            if (reinforcement(chosenAction, label) == 1)  // Has the system learned the behaviour that was positively reinforced?
            {
                ++goodDecisions;
            }
            else
            {
               // std::cout << "Bad training action " << (int) chosenAction << " on numeral " << (int) label << std::endl;
                ++confusion[(int)chosenAction * 10 + label];
                ++countConfusion;
            }
        }
    } // Loop  over training images
    auto test_finished = std::chrono::steady_clock::now();
    double ProbabilityOfCorrect = (double)goodDecisions / 60000.0;
    if (lowSumWeightsCount > 0) std::cout << "Low sum weights count " << lowSumWeightsCount << "." << std::endl;
    auto elapsed_test = std::chrono::duration_cast<std::chrono::milliseconds>(test_finished - start_test_set);
    std::cout << "Mean time required to classify an image           " << elapsed_test.count() / 60000.0 << " milliseconds. " << std::endl;
    std::cout << "Probability of a correct decision on TRAINING DATA " << ProbabilityOfCorrect * 100.0 << " percent " << std::endl;
    std::cout << "Error matrix. Rows are actions 0 to 9 ";
    for (int i = 0; i < 100; ++i)
    {
        if (i % 10 == 0) std::cout << std::endl;
        std::cout << confusion[i] << " ";
    }
    std::cout << "\nTotal mistakes testing on TRAINING DATA " << countConfusion << " or " << countConfusion / 600.0 << " % ";
 } // End of testOnTRAINING_SET

void view(mnist_image_t* pimage, uint8_t label)
{
    for (uint32_t i = 0; i < nSensors; ++i)
    {
        if (pimage->pixels[i] > 5) // TBD
        {
            std::cout << "M";
        }
        else
        {
            std::cout << " ";
        }

        if (i % 28 == 27)
        {
            std::cout << std::endl;
        }
        if (i == nSensors - 1)
        {
            int ilabel = (int)label;
            std::cout << "The above numeral has label " << ilabel << std::endl;
        }
    }
} // End of view