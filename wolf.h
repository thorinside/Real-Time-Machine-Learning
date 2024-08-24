/* wolf.h
Copyright (C) 2024 William Ward Armstrong
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
#pragma once
#include <memory>
#include <limits>
#include <vector>
#include <array>
#include <assert.h>
#include <iostream>
// WOLF stands for Weighted Overlapping Linear Functions, the approximation method used.
extern int numberLF_trees;
extern int minSamples2Split;
extern int constancyLimit ;
extern double convolutionRadius;
extern size_t treeNo;
extern double reinforcement(uint8_t action, uint8_t label);
long leafCount{0}; // Counts the number of leaf Sample Blocks (SB), i.e. nodes which compute probabilities and are not used for branching
bool treeFinal{ false }; // Signals the end of tree growth.
size_t non_ID{0x3777777777 };
extern mnist_image_t* train_images;
extern uint8_t* train_labels;
std::vector<double> initC{}; // Each tree has this starting centroid of all samples which may then
// shift a bit so the samples in the blocks become different.
std::vector<size_t> image_numbers_setup{}; // Aids speed.
std::array <double, 49> kernel; // A convolution kernel.
     // Comnvolution may improve results on the test set with no speed penalty.
void initCentroidSampleNumbers(int nSensors);
// In what follows, S (D) means less (greater than or equal to) the features-computed value.
// One can think of it as to the left or right on the axis. Latin: Sinister = left, Dexter = right.

struct SF // Used for features
{
    size_t sensor;
    double factor;
};

struct SB  // The Sample Blocks are of two types: 1. feature nodes that partition sample blocks into two forming a tree
           // and 2. leaf blocks with linear functions as membership functions giving the probability of the target class.
           // When a leaf block is split in two by a feature, it becomes a feature block with the two resulting blocks as leaf children.
{
    bool   is_final {false};                          /* Initially false; true when SB can't change/split   */
    std::vector<double> C{};                          /* Centroid vector, C[nSensors] is output centroid    */
    std::vector<double> W{};                          /* Weight vector for active domain components         */
    std::vector<size_t> active;                       /* list of pixels deemed non-constant on a block      */
    std::vector<size_t> image_numbers{};              /* a list of indices of images in train_images        */
    std::vector<SF>  features;                        /* A feature vector is at a node of the feature tree  */
    double FTvalue{};                                 /* Value used at a branch of a feature tree           */
    size_t FTS {non_ID};                              /* Left SBid stored at a branching of a feature tree   */
    size_t FTD{non_ID};                               /* Right SBid at a feature tree branching              */
    size_t FTparent{ non_ID };                        /* The parent of this block in the feature tree        */
 }; 

class cLF_tree
{
    // Data members
    size_t m_treeNo{ 0 };                  // The number of the current tree. Its associated action is  m_treeNo % 10 when sed for MNIST classification;
    uint32_t m_nSensors{784};              // Number of sensors (pixels) of MNIST data
    std::vector<SB> m_SB{};                // Vector to contain all SB structs

public: // Methods
    // We first create one leaf SB at index 0 of the m_SB vector which is to contain all SB nodes.
    // At the first split, it will become a feature node and two leaves will be its children in the Learning Feature Tree.
    cLF_tree* create_LF_tree();
    void    setTreeNumber(size_t treeNo);
    void    setSensorNumber(uint32_t nSensors); 
    void    loadSBs(long numberImages);
    void    addTRsample(mnist_image_t* images, size_t imageNo);
    size_t  createSB();
    void    growTree();
    void    createFeature(size_t SBid);
    void    splitSB(size_t SBid, int c);
    size_t  findBlock( mnist_image_t* pX, double& weight);
    size_t  findBlock(mnist_image_t* pX);
    double  evalBoundedWeightedSB(mnist_image_t* pimage, double& weight);
    double  evalBoundedSB(mnist_image_t* pimage);
    void    checkFinalTree();
    void    makeSBfinal(size_t SBid);
    void    makeFTnode(size_t SBid);
    void    convolution( std::vector<SF>& featureVector);
};

cLF_tree* create_LF_tree()
{
    // A Learning Feature tree partitions the training samples in several stages so two classes, targets
    // and non-targets, tend to be concentrated in different blocks. Each partition of a block into two
    // blocks is based on a linear function of sample values. At the leaves of the Feature Tree are
    // linear functions which estimate the probability of the target class at each point of the block.
    //  Classification accuracy is improved by using several trees on the same problem.
    // Learning Feature Trees grow from a single leaf to hundreds or thousands of leaves.
    // They are independent from each other and can grow in parallel on large enough hardware.
    // Evaluation on a new sample is expected to be fast enough for real-time systems.
    auto pLF_tree = new cLF_tree;
    pLF_tree->createSB();
    uint32_t sensor = 0;
    return pLF_tree;
}

void    cLF_tree::setTreeNumber(size_t treeNo)
{
    cLF_tree::m_treeNo =treeNo;
}

void   cLF_tree::setSensorNumber(uint32_t nSensors)
{
    cLF_tree::m_nSensors = nSensors;
}

void cLF_tree::loadSBs( long numberImages)
{
    static long loadFirstImage{ 0 }; 
   // std::cout << "Loading the SB block(s) with " << numberImages << " image numbers. " << std::endl;
    size_t imageNo = 0;
    long samplesAdded = 0;
    for (size_t imageCount = 0; imageCount < numberImages; ++imageCount)
    {
        imageNo = (loadFirstImage + imageCount) % 60000;
        addTRsample(train_images + imageNo, imageNo);
    }
    loadFirstImage = (loadFirstImage + numberImages) % 60000; // Image with which the next image-number load into SBs begins
} // End of loadSBs

void cLF_tree::addTRsample(mnist_image_t* pImage, size_t imageNo)
{
    double weight; // The weight is important when the image is evaluated on the test set.
    size_t SBid = findBlock(pImage, weight);
    m_SB[SBid].image_numbers.push_back(imageNo);
} // End of addTRsample

size_t  cLF_tree::createSB() // The splitting SB at pointer pSB leads to creation and initialization of a new SB
{
    uint32_t nSensors = m_nSensors;
    size_t SBid = m_SB.size();               // Get the unique identifier, or index, of a sample block (SB).
    m_SB.push_back(SB());                    // Add the new SB to the vector m_SB of all SBs (including those used as feature tree nodes)
    m_SB[SBid].is_final = false;             // A final SB does not change any more
    m_SB[SBid].C.assign(nSensors + 1, 0);    // The centroid ( an engram) of the images on the block, a vector including the output centroid at index nSensors
    m_SB[SBid].W.assign(nSensors + 1, 0);    // The weights on nSensors inputs of the linear function on a block. The W indices, like those of C, don't change due to inactivity of sensors
    m_SB[SBid].FTS = non_ID;                 // non_ID is a huge size_t value that indicates a not initialized left branch of a feature tree. (S = sinister in Latin = left ).
    m_SB[SBid].FTD = non_ID;                 // Indicates the above for the right branch. (D = dexter in Latin)
    m_SB[SBid].FTvalue = 0;                  // This is the split value at a node of the feature tree, the value of the SB's feature at the centroid vector or other placement.
    m_SB[SBid].active.clear();
    if (SBid == 0)
    {
        for (uint32_t sensor = 0; sensor < nSensors; ++sensor) // Sets the initially nSensors active sensors of SBid = 0 to 0,1,2 ...nSensors - 1 
            // Others get activity set during a split.
        {
            m_SB[SBid].active.push_back(sensor);
        }
    }
    ++leafCount; // Creates one additional leaf SB
    return SBid;
} // End of createSB

void initCentroidSampleNumbers(int nSensors) //sets up two vectors: initC and image_numbers_setup that initialize trees
{
    mnist_image_t* pimage = nullptr;
    double accuC = 0;
    initC.assign(nSensors + 1, 0);
    for (size_t sensor = 0; sensor < nSensors; ++sensor)
    {
        accuC = 0;
        for (size_t imageNo = 0; imageNo < 60000; ++imageNo)
        {
            accuC += (double)(train_images + imageNo)->pixels[sensor];
        }
        initC[sensor] = accuC / 60000.0; 
    }
    for (size_t s = 0; s < 60000; ++s) image_numbers_setup.push_back(s);
    // initC and image_numbers_setup will speed up initialization of SampleBlock SBid = 0 of all trees
} // End of initCentroidSampleNumbers

void cLF_tree::growTree()
{
    if (m_SB.size() == 1)
    {
        treeFinal = false;
        m_SB[0].image_numbers = image_numbers_setup;
    }
    // This routine does four things with image and label data in collaboration with two other routines: createFeature and splitSB.
    // 1. It takes the precomputed centroid of domain dimensions for SB 0 and adds some shifts. The centroids of other SB are computed in splitSB.
    // 2. It determines the variances of the image samples' active component dimensions to support step 3.
    // 3. It removes some domain dimensions whose values vary little in the samples of the block and are "deemed constant" over an SB block, 
    // 4. It finds the weights of linear functions with domain SB to fit the data based on the remaining active dimensions.
    uint32_t nSensors = m_nSensors; // This is the number of domain coordinates (number of pixels in an image = size of retina)
    size_t sensor{ 0 };
    size_t sensorIndex{ 0 };
    mnist_image_t* pimage = nullptr;
    size_t SBid{ 0 };
    size_t imageNo{ 0 }; // This is the index of the image and its label in the MNIST training data files of images and labels
    size_t local_imageNo{ 0 }; //This is the index of image numbers (imageNo values) which are stored with the SB.
    double  y{ 0 }; // y is the reinforcement for the given action and image label.
    std::vector<size_t> stayActive{}; // This temporarily records the remaining active sensors prior to elimination of some from the active list
    double accu = 0;
    for ( SBid = 0; SBid < m_SB.size(); ++SBid)  // for ( SBid = 0; SBid < size_m_SB; ++SBid) uses iterations
    {
        if (m_SB[SBid].is_final == false && m_SB[SBid].image_numbers.size() > 0)
        {
            if (SBid == 0) // We compute a tree-shifted centroid for the domain coordinates of SB 0, others are done during splitSB
            {
                m_SB[0].features.clear();
                m_SB[0].C = initC; // presomputed to save time
                for (sensor = 0; sensor < nSensors; ++sensor)
                {
                    m_SB[SBid].C[sensor] *= (0.5 + (double)m_treeNo / (double)numberLF_trees); // multiplicative shift keeps things positive even for pixels near 0 intensity
                    // This shift of the centroid should make all the trees used in a weighted average different
                }
            }
            // In general, an SB can't split if it has too few image samples
            // We have to take the following steps in all cases so the piece can be made final if necessary.
            // we store some values important for computing variance and doing least squares to get weights W
            double En = 0;
            double Ex = 0;
            double Ex2 = 0;
            double Exy = 0;
            double Ey = 0;
            // Compute the mean reinforcement for this SB using its local image numbers
            local_imageNo = 0;
            while (local_imageNo < m_SB[SBid].image_numbers.size())
            {
                imageNo = m_SB[SBid].image_numbers[local_imageNo];
                Ey += reinforcement((uint8_t) m_treeNo % 10, train_labels[imageNo]); // ?? this may only be correct for classifying all digits
                ++local_imageNo;
            }
            En = (double) m_SB[SBid].image_numbers.size();
            stayActive.clear(); // Helps to change the activity vector
            long value{ 0 }; // sum of sensor values
            // We now compute the "E values" necessary for solving least squares by the normal equations
            sensorIndex = 0;
            while (sensorIndex < m_SB[SBid].active.size())
            {
                sensor = m_SB[SBid].active[sensorIndex];
                local_imageNo = 0;
                while (local_imageNo < m_SB[SBid].image_numbers.size())
                {
                    imageNo = m_SB[SBid].image_numbers[local_imageNo];
                    pimage = train_images + imageNo;
                    y = reinforcement(m_treeNo % 10, train_labels[imageNo]);
                    {
                        value = pimage->pixels[sensor];
                        // Ex += value; This divided by En just recalculates the centroid.
                        Ex2 += value * value;
                        Exy += value * y;
                    }
                    ++local_imageNo;
                }
                // Take the means
                Ex = m_SB[SBid].C[sensor];
                Ex2 /= En;
                Exy /= En;
                Ey /= En;
                m_SB[SBid].C[nSensors] = Ey; // This is the mean value of the output component of the centroid for this SB
                // Compute the mean value and variance of intensity for each active sensor. 
                // Low variance entails making the sensor inactive on the SB ( and hence on all results of splitting SB).
                // An inactive sensor doesn't participate in features of a block.
                // Determine which variables are deemed constant over this SB (by the constancy criterion).
                double Variance = Ex2 - Ex * Ex;
                if (Variance > constancyLimit * constancyLimit) // Variance test: we need following values only for sensors that aren't deemed constant on the SB block.
                {
                    stayActive.push_back(sensor); // Record the fact that this sensor is not considered constant
                    // If this SB can't split, we need the weights of the active sensors
                    m_SB[SBid].W[sensor] = (Exy - Ex * Ey) / Variance;
                }
                ++sensorIndex;
            }
            m_SB[SBid].active.clear(); // Remove the constant sensors on block SBid starting by clearing the active vector
            size_t j = 0;
            while (j < stayActive.size())
            {
                m_SB[SBid].active.push_back(stayActive[j]);
                ++j;
            }
        } // End if SBid is_final = false && m_SB[SBid].image_numbers.size() > 0 )condition
        if (m_SB[SBid].image_numbers.size() < minSamples2Split)
        {
            if (m_SB[SBid].image_numbers.size() == 0) m_SB[SBid].C[nSensors] = 0;
            makeSBfinal(SBid);
        }
        else
        {
            createFeature(SBid);
        }
    } // End of for ( SBid = 0; SBid < size_m_SB; ++SBid)
} // End of growTree

void cLF_tree::createFeature(size_t SBid)
{
    // Precondition: m_SB[SBid].image_numbers.size() >= 2 * minSamples2Split && 9 * m_SB[SBid].image_numbers.size() >= 2 * m_SB[SBid].active.size()
    // This is a multi-level feature extraction part of the algorithm. By appropriate splitting, it tries to concentrate target-
    //  and non-target images on different children of split SB. This automatic feature construction needs much further work.
    // For example, we can choose the position of the split so that all targets are all on the D side, i.e. they have greater
    // values than some threshold. If this is the case, there are no targets on the S side and
    //  the S side is made final with function constant 0 and no active sensors if the number of imange samples is sufficient.
    // To start, we need a rough idea of the distribution of targets and non-targets on the SB on each sensor.
    // We find the upper and lower bounds of each set for active sensors.(Inactive sensors don't have enough range to discriminate.).
    int nSensors = m_nSensors;
    size_t activeSize = m_SB[SBid].active.size();
    size_t sensor{ 0 };
    mnist_image_t* pimage = nullptr;
    size_t imageNo{ 0 };
    long countTargets = 0;
    long countNonTargets = 0;
    size_t local_imageNo = 0;
    while (local_imageNo < m_SB[SBid].image_numbers.size())
    {
        imageNo = m_SB[SBid].image_numbers[local_imageNo];
        if (reinforcement(m_treeNo % 10, train_labels[imageNo]) == 1) // target class
        {
            ++countTargets;
        }
        else
        {
            ++countNonTargets;
        }
        ++local_imageNo;
    }
    if (countNonTargets == 0)
    {
        makeSBfinal(SBid);
        m_SB[SBid].C[nSensors] = 1.0; // All images on SBid are targets
        return;
    }
    if (countTargets == 0)
    {
        makeSBfinal(SBid);
        m_SB[SBid].C[nSensors] = 0; // All images on SBid are non_targets
        return;
    }
    // Both are non-zero, so we have a mixture of targets and non-targets
    // We look for sensors that show a difference between distributions of targets and non-targets
    double fltCountT = countTargets;
    double fltCountN = countNonTargets;
    m_SB[SBid].C[nSensors] = fltCountT / (fltCountT + fltCountN); // This can be enhanced by weights calculated previously
    // We collect statistics using upper and lower bounds on both targets and non-targets
    std::vector<int> tSensor; // for computing centroids of intensity of targets for sensors
    tSensor.assign(nSensors, 0);
    std::vector<int> nontSensor; // same for non-targets
    nontSensor.assign(nSensors, 0);
    size_t sensorIndex = 0;
    while (sensorIndex < activeSize) // Only active sensors have enough variation to discriminate
    {
        sensor = m_SB[SBid].active[sensorIndex];
        local_imageNo = 0;
        while (local_imageNo < m_SB[SBid].image_numbers.size())
        {
            imageNo = m_SB[SBid].image_numbers[local_imageNo];
            pimage = train_images + imageNo;
            if (reinforcement(m_treeNo % 10, train_labels[imageNo]) == 1) // targets
            {
                tSensor[sensor] += (long)pimage->pixels[sensor];
            }
            else
            {
                nontSensor[sensor] += (long)pimage->pixels[sensor];
            }
            ++local_imageNo;
        }
        // tSensor[sensor] is the total intensity of that sensor (pixel) for all the training *target* images on SB.
        // nontSensor[sensor] is for the non-targets..
        ++sensorIndex;
    }
    // There is at least one target and one non-target image as a result of the first step above.
    // We take the sensor of maximal difference of centroids for sure and some others of lesser difference.
    m_SB[SBid].features.clear();
    double sensorDiff = 0;
    double maxSD = -FLT_MAX;
    double minSD = FLT_MAX;
    // get the maximum difference of centroids over all active semsors
    sensorIndex = 0;
    while (sensorIndex < activeSize)
    {
        sensor = m_SB[SBid].active[sensorIndex];
        // Examine the target and non-target centroids of image values on the active sensors
        sensorDiff = tSensor[sensor] / (double)countTargets - nontSensor[sensor] / (double)countNonTargets;
        if (maxSD < sensorDiff) maxSD = sensorDiff;
        if (minSD > sensorDiff) minSD = sensorDiff;
        ++sensorIndex;
    }
    // Pick out features                                                                  
    int featureCount = 0;
    sensorIndex = 0;
    SF sf;
    while (sensorIndex < activeSize)
    {
        sensor = m_SB[SBid].active[sensorIndex];
        sensorDiff = tSensor[sensor] / (double)countTargets - nontSensor[sensor] / (double)countNonTargets;
        if (sensorDiff >= 0.05 * maxSD ) 
        {
             sf.sensor = sensor; sf.factor = sensorDiff/(double) maxSD;
            m_SB[SBid].features.push_back(sf); // Form the features vector, which is a vector of sensors (as ints).
            ++featureCount;
        }
        if (  sensorDiff <= 0.05 * minSD)
        { 
            sf.sensor = sensor; sf.factor = -sensorDiff / (double) minSD;//This should make the factor negaive
            m_SB[SBid].features.push_back(sf); // Form the features vector, which is a vector of sensors (as ints).
            ++featureCount;
        }
        ++sensorIndex;
    }
    tSensor.clear();
    nontSensor.clear(); 
    // Normalize the feature vector any change of a single image pixel by one unit changes the 
    // feature by no more than one unit
    double maxfactor = 0;
    for (auto sf : m_SB[SBid].features) if( fabs(sf.factor) > maxfactor) maxfactor = fabs(sf.factor);
    for (auto sf : m_SB[SBid].features) sf.factor /= maxfactor;
    convolution(m_SB[SBid].features); // Convolve the feature vector with a kernel
    // Mow we have the features vector. Let's see how it works.
    double featureSum; // Sum of all the sensor values of an image for sensors in the feature vector.
    double minTFS = FLT_MAX; // minimum Target Feature Sum below all targets(on this SB)
    double maxTFS = -FLT_MAX; // maximum Target Feature Sum above all targets (on this SB)
    double minNFS = FLT_MAX; // minimum Non-Target Feature Sum below all non-targets(on this SB)
    double maxNFS = -FLT_MAX; // maximum Non-Target Feature Sum above all non-targets (on this SB)
    local_imageNo = 0;
    while (local_imageNo < m_SB[SBid].image_numbers.size())
    {
        imageNo = m_SB[SBid].image_numbers[local_imageNo];
        pimage = train_images + imageNo;
        featureSum = 0;
        for (auto sf : m_SB[SBid].features) featureSum += pimage->pixels[sf.sensor] * sf.factor;
        if (reinforcement( m_treeNo % 10, train_labels[imageNo]) == 1)
        {
            if (featureSum < minTFS) minTFS = featureSum;
            if (featureSum > maxTFS) maxTFS = featureSum;
        }
        else
        {
            if (featureSum < minNFS) minNFS = featureSum;
            if (featureSum > maxNFS) maxNFS = featureSum;
        }
        ++local_imageNo;
    }
    // Shift for SBid = 0 makes trees differ
    if (SBid == 0)
    {
        minTFS -= m_treeNo; minNFS -= m_treeNo; maxTFS += m_treeNo; maxNFS += m_treeNo;
    }



    long tbminNFS = 0;
    long tamaxNFS = 0;
    long nbminTFS = 0;
    long namaxTFS = 0;
    local_imageNo = 0;
    while (local_imageNo < m_SB[SBid].image_numbers.size())
    {
        imageNo = m_SB[SBid].image_numbers[local_imageNo];
        pimage = train_images + imageNo;
        featureSum = 0;
        for (auto sf : m_SB[SBid].features) featureSum += pimage->pixels[sf.sensor] * sf.factor;
        {
            // We can calculate exactly for each SB
            // 1. The number tbminNFS of targets with feature value below minNFS
            // 2. The number tamaxNFS of targets with feature value above maxNFS
            // 3. The number nbminTFS of non-targets with feature value below minTFS
            // 4. The number namaxTFS of non_targets with feature value above maxTFS
            // We choose a threshold corresponding to the maximum number of images in the groups 1 to 4.
            if (reinforcement( m_treeNo % 10, train_labels[imageNo]) == 1)
            {
                // Targets
                if (featureSum  < minNFS)
                {
                    ++tbminNFS;
                }
                if (featureSum > maxNFS)
                {
                    ++tamaxNFS;
                }
            }
            else
            {
                // Non-Targets        
                if (featureSum < minTFS)
                {
                    ++nbminTFS;
                }
                if (featureSum > maxTFS)
                {
                    ++namaxTFS;
                }
            }
        }
        ++local_imageNo;
    }
    // We need a struct to find which of the above cases has a maximal sized set of images above the minimum size.
    struct cv { int c; long v; };
    cv cv1{ 1, tbminNFS };  cv cv2{ 2, tamaxNFS }; cv cv3{ 3, nbminTFS }; cv cv4{ 4, namaxTFS };
    cv ans1 = (cv1.v > cv2.v) ? cv1 : cv2; cv ans2 = (cv3.v > cv4.v) ? cv3 : cv4;
    cv ans = (ans1.v > ans2.v) ? ans1 : ans2;
    if (ans.v < minSamples2Split || m_SB[SBid].image_numbers.size() - ans.v < minSamples2Split) { ans.c = 5; } // Case 5 splits the piece in two at a mid-point of all four threshold values
    switch (ans.c)
    {
    case 1: m_SB[SBid].FTvalue = minNFS; break;
    case 2: m_SB[SBid].FTvalue = maxNFS; break;
    case 3: m_SB[SBid].FTvalue = minTFS; break;
    case 4: m_SB[SBid].FTvalue = maxTFS; break;
    case 5:/* featureSum = 0; // Another way of choosing FTvalue
            for (SF sf : m_SB[SBid].features)
            {
                featureSum += sf.factor * m_SB[SBid].C[sf.sensor];
            }
            m_SB[SBid].FTvalue = featureSum;*/
            m_SB[SBid].FTvalue = (minNFS + maxNFS + minTFS + maxTFS)/4;
}
    // std::cout << "Case " << ans.c << " threshold " << m_SB[SBid].FTvalue;
    // In case 1, the S part is made final and its C[nSensors]  = 1.0,
    // in case 2, the D part is made final and its C[nSensors] = 1.0,
    // in case 3, the S part is made final and its C[nSensors] = 0,
    // in case 4, the D part is made final and its C[nSensors] = 0.
    // in case 5, neither part is made final and its C[nSensors] is not set in splitSB.
    splitSB(SBid, ans.c);
} // End of createFeature

void cLF_tree::splitSB(size_t SBid, int cxx)
{
    // Splitting adds two SBs,  but removes one leaf SB which now does branching
    int nSensors = m_nSensors;
    size_t SSBid = createSB(); // Create two child SBs N.B. Before this, we have to make sure they will each have enough images.
    size_t DSBid = createSB();
    --leafCount; // We created two leaf SBs, but changing the leaf parent to a branching node of the LF-tree removes one leaf SB
    // Set the children's activity like SBid's activity
    size_t sensorIndex = 0;
    size_t sensor = 0;
    // Copy the activity vector of SBid to the children
    while (sensorIndex < m_SB[SBid].active.size())
    {
        sensor = m_SB[SBid].active[sensorIndex];
        m_SB[SSBid].active.push_back(sensor);
        m_SB[DSBid].active.push_back(sensor);
        ++sensorIndex;
    }
    // Assign the image numbers to the S (left) or D (right) child
    size_t imageNo = 0;
    // Compute the feature values of the images whose numbers are on SBid for distribution of the numbers to the children
    m_SB[SSBid].C.assign(nSensors + 1, 0); // Make room for the output centroid
    m_SB[DSBid].C.assign(nSensors + 1, 0);
    int Sn = 0; int Dn = 0;
    size_t local_imageNo = 0;
    while (local_imageNo < m_SB[SBid].image_numbers.size())
    {
        imageNo = m_SB[SBid].image_numbers[local_imageNo];
        // compute the feature value of the image
        double featureSum = 0;
        for (auto sf : m_SB[SBid].features)  featureSum += (double)train_images[imageNo].pixels[sf.sensor] * sf.factor;
        // Use the feature to compute the centroids of the children and to distribute the image numbers to the children
        sensor = 0;
        while (sensor < nSensors)
        {
            if (featureSum < m_SB[SBid].FTvalue)
            {
                ++Sn;
                m_SB[SSBid].C[sensor] += (double)train_images[imageNo].pixels[sensor];
            }
            else
            {
                ++Dn;
                m_SB[DSBid].C[sensor] += (double)train_images[imageNo].pixels[sensor];
            }
            ++sensor;
        }
        if (featureSum < m_SB[SBid].FTvalue)
        {
            // put the image index in train_images on the S-side images
            m_SB[SSBid].image_numbers.push_back(imageNo);
        }
        else
        {
            // put the image number on the D-side.
            m_SB[DSBid].image_numbers.push_back(imageNo);
        }
        ++local_imageNo;
    }
    sensor = 0;
    while (sensor < nSensors)
    {
        m_SB[SSBid].C[sensor] /= Sn; // S-side centroid
        m_SB[DSBid].C[sensor] /= Dn; // D-side centroid
        ++sensor;
    }
    m_SB[SBid].image_numbers.clear();
    makeFTnode(SBid); // Only the leaves of the feature tree will be processed
    m_SB[SSBid].is_final = false;
    m_SB[DSBid].is_final = false;
    // In case 1, the S part is made final and its C[nSensors]  = 1.0,
    // in case 2, the D part is made final and its C[nSensors] = 1.0,
    // in case 3, the S part is made final and its C[nSensors] = 0,
    // in case 4, the D part is made final and its C[nSensors] = 0.
    switch (cxx)
    {
    case 1: m_SB[SSBid].is_final = true; m_SB[SSBid].C[nSensors] = 1.0; break;
    case 2: m_SB[DSBid].is_final = true; m_SB[DSBid].C[nSensors] = 1.0; break;
    case 3: m_SB[SSBid].is_final = true; m_SB[SSBid].C[nSensors] = 0; break;
    case 4: m_SB[DSBid].is_final = true; m_SB[DSBid].C[nSensors] = 0; break;
    case 5:;
    }
    // Set the parent and child indices in the LF-tree for branching
    m_SB[SSBid].FTparent = m_SB[DSBid].FTparent = SBid;
    m_SB[SBid].FTS = SSBid;
    m_SB[SBid].FTD = DSBid;
}  // End of splitSB

size_t cLF_tree::findBlock( mnist_image_t* pX, double& weight)
{
    weight = 1.0;
    size_t Blkid =0;
    while (true)
    {
        if (m_SB[Blkid].FTS == non_ID)
        {
            return Blkid; // Blkid is the leaf sought
        }
        else
        {
            // Blkid is not the goal of the search
            // Compute the feature for the Blkid feature tree node
            double featureSum = 0;
            for (auto sf : m_SB[Blkid].features)
            {
                featureSum += pX->pixels[sf.sensor] * sf.factor;
            }
            double dist = featureSum - m_SB[Blkid].FTvalue;
            Blkid = (dist < 0) ? m_SB[Blkid].FTS : m_SB[Blkid].FTD;
            dist = fabs(dist);
            weight = dist < weight ? dist : weight;
        }
    }
} // End of findBlock(size_t BlockID, mnist_image_t* pX)

size_t cLF_tree::findBlock( mnist_image_t* pX) // Faster version without weight
{
    size_t Blkid = 0;
    while (true)
    {
        if (m_SB[Blkid].FTS == non_ID)
        {
            return Blkid; // Blkid is the leaf sought
        }
        else
        {
            // Blkid is not the goal of the search
            // Compute the feature for the Blkid feature tree node
            double featureSum = 0;
            for (auto sf : m_SB[Blkid].features)
            {
                featureSum += pX->pixels[sf.sensor] * sf.factor;
            }
            double dist = featureSum - m_SB[Blkid].FTvalue;
            Blkid = (dist < 0) ? m_SB[Blkid].FTS : m_SB[Blkid].FTD;
        }
    }
}

double cLF_tree::evalBoundedWeightedSB(mnist_image_t* pimage, double& weight)
{
    // Evaluates the bounded weighted function of an SB for the given image and returns the weight as a reference
    int nSensors = m_nSensors;
    size_t SBid = findBlock( pimage, weight);
    double value = m_SB[SBid].C[nSensors];
    size_t sensorIndex = 0;
    size_t sensor = 0;
    while (sensorIndex < m_SB[SBid].active.size())
    {
        sensor = m_SB[SBid].active[sensorIndex];
        value += m_SB[SBid].W[sensor] * ((double)(pimage->pixels[sensor]) - m_SB[SBid].C[sensor]);
        ++sensorIndex;
    }
    if (value < 0.0) return 0; // We could replace these three lines by a sigmoid function
    if (value >= 1.0) return weight;
    return value * weight;
} //End of evalBoundedWeightedSB

double cLF_tree::evalBoundedSB(mnist_image_t* pimage) // For evaluating individual trees on the training set
{
    // evaluates the linear function of the block
    int nSensors = m_nSensors;
    size_t SBid = findBlock(pimage);
     double value = m_SB[SBid].C[nSensors];
     size_t sensorIndex = 0;
    size_t sensor = 0;
        while (sensorIndex < m_SB[SBid].active.size())
        {
            sensor = m_SB[SBid].active[sensorIndex];
            value += m_SB[SBid].W[sensor] * ((double)(pimage->pixels[sensor]) - m_SB[SBid].C[sensor]);
            ++sensorIndex;
        }
        if (value < 0.0) return 0; // We could replace these three lines by a sigmoid function
        if (value >= 1.0) return 1.0;
        return value;
} //End of evalBoundedSB

void cLF_tree::makeSBfinal(size_t SBid)
{
    m_SB[SBid].is_final = true;
    m_SB[SBid].features.clear();
    if( m_SB[SBid].image_numbers.size() < m_SB[SBid].active.size()) m_SB[SBid].active.clear();
    m_SB[SBid].image_numbers.clear();
}

void cLF_tree::makeFTnode(size_t SBid) // After splitting, a non-final leaf becomes a feature tree node. The feature vector and FTvalue are kept.
{
    m_SB[SBid].is_final = true;
    m_SB[SBid].image_numbers.clear();
}

void cLF_tree::checkFinalTree()
{
    double correctLevel = 0.5 ;
    long numberTarget = 0;
    long numberNonTarget = 0;
    long numberRightTarget = 0;
    long numberRightNonTarget = 0;
    uint8_t label = 0;
    double value;
    size_t imageNo = 0;
    while (imageNo < 60000)
    {
        value = evalBoundedSB(train_images + imageNo);
        label = train_labels[imageNo];
        if (label == m_treeNo % 10)
        {
            ++numberTarget;
            if (value >= correctLevel)  ++numberRightTarget;  // Changedfrom 0.5 to look better
        }
        else
        {
            ++numberNonTarget;
            if (value < correctLevel) ++numberRightNonTarget;
        }
        ++imageNo;
    } // End loop over all the SB
    // Since this is over all training images, there is no possibility of denominators being 0
    if (numberNonTarget > 0) std::cout << "Correct out of " << numberNonTarget << " non targets " <<
    numberRightNonTarget << "                 " << 100.0f * (double)numberRightNonTarget / (double)numberNonTarget << "%" << std::endl;
    if (numberTarget > 0)std::cout << "Correct out of " << numberTarget << " targets " <<
    numberRightTarget << "                      " << 100.0f * (double)numberRightTarget / (double)numberTarget << "%" << std::endl;
} // End of checkFinalTree

void createKernel()
{
    double rho = (convolutionRadius < 3.1) ? 0 : 1.0; // First a round kernel; then a horizontal one
    double d= 0;
    double accu = 0;
    for (int h = 0; h < 7; ++h)
    {
        for (int v = 0; v < 7; ++v)
        {
            d = sqrt((h - 3) * (h - 3) + (v - 3) * (v - 3))/ convolutionRadius;
            kernel[7 * v + h] =   (1.0 - rho)  *  (d < 1.0) ? (1.0 - 3.0 * d * d + 2.0 * d * d * d) : 0;
            kernel[7 * v + h] +=   rho * (d < 1.0 && v == 3) ? (1.0 - 3.0 * d * d + 2.0 * d * d * d) : 0;
            accu += kernel[7 * v + h];
        }
    }
    std::cout << "\nKernel for this run\n";
    // Make the weights of the kernel sum to 1.0
    for (int i = 0; i < 49; ++i)
    {
        kernel[i] /= accu;
        if (i % 7 == 0) std::cout << std::endl; std::cout << kernel[i] << " ";
    }
    std::cout << std::endl;
} // End of createKernel

void cLF_tree::convolution( std::vector<SF>& featureVector)
{
    std::vector<SF> fV = featureVector; // fV is a copy of the original featureVector
    SF sf0{ 0,0 };
    featureVector.assign(784, sf0);
    size_t k = 0;
    double value = 0;
    for (int j = 0; j < fV.size();++j)
    {
        // We have to add weight to featureVector at sensors around the current sensor of fV
        // We use a seven x seven convolution kernel
        if (fV[j].factor == 0) continue;
        size_t H = fV[j].sensor % 28;
        size_t V = fV[j].sensor / 28;
        for (int h = 0; h < 7; ++h)
        {
            for (int v = 0; v < 7; ++v)
            {
                if (kernel[7 * v + h] == 0) continue;
                // the updated component is at H - 3 + h, V - 3 + v or in terms of the featureVector
                k = 28 * (V - 3 + v) + (H - 3 + h);
                if (H - 3 + h >= 0 && H - 3 + h < 28 && V - 3 + v >= 0 && V - 3 + v < 28)
                {
                    featureVector[k].sensor = k;
                    featureVector[k].factor += fV[j].factor * kernel[7 * v + h];
                }
            }
        }
    }
    //We can purge from featureVector the factor zero entries
    fV = featureVector;
    featureVector.clear();
    for (int j = 0; j < fV.size(); ++j)
    {
        if (fV[j].factor != 0) featureVector.push_back(fV[j]);
    }
}
