#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <string>
#include <math.h>
#include <map>

#define CLASS_COUNT 3

class BayesClassifier
{
public:
    BayesClassifier(int argc, char** argv);
    BayesClassifier();
    void fit(std::vector<std::pair<std::string, int>>& data);
    int predict(std::string& text);
    void predict(std::vector<std::pair<std::string, int>>& data);
    void predict(std::string& stats_file, std::string& input_file, std::string& output_file);
    double getAccurancy();
    std::vector<long double> getPrecision();
    std::vector<long double> getRecall();
    std::vector<long double> getF1Measure();
private:
    std::string output_file;
    int answers = 0;
    std::vector<std::string> tags = {
        "Cats",
        "Dogs",
        "Hamsters"
    };
    int right_answers = 0;
    long double right_tag_answers = 0;
    std::vector<int> class_predicted_answers;
    std::vector<int> class_right_answers;
    std::vector<int> all_class_answers;
    void preprocessing(std::vector<std::pair<std::string, int>>& data);
    void transform(std::string& word);
    void stats_preprocessing(std::string& stats_file);
    std::map<std::string, std::vector<long double>> softmax(std::map<std::string, std::vector<long double>>& probs);
    std::vector<std::pair<std::string, std::string>> input_preprocessing(std::string& input_file);
    void get_data(const std::string& filename);
    void fit(std::vector<std::pair<std::string, std::vector<int>>>& data);
    std::vector<int> GetClass(std::vector<std::string>& ClassNames);
    std::vector<std::string> GetClass(std::vector<int>& ClassNames);
    long double class_belong_rate = 0.33;
    std::vector<int> obj_count;
    std::vector<std::map<std::string, long long>> word_frequency;
    int training_size;
    int classify_size;
    int c0_obj_count = 0;
    int c1_obj_count = 0;
    std::map<std::string, long long> frequency0;
    std::map<std::string, long long> frequency1;
    std::map<std::string, std::vector<int>> test_answers;
};