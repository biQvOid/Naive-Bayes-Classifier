#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <string>
#include <map>

class BayesClassifier
{
public:
    void fit(std::vector<std::pair<std::string, int>>& data);
    int predict(std::string& text);
    void predict(std::vector<std::pair<std::string, int>>& data);
    double getAccurancy();
private:
    int answers = 0;
    int right_answers = 0;
    void preprocessing(std::vector<std::pair<std::string, int>>& data);
    int training_size;
    int classify_size;
    int c0_obj_count = 0;
    int c1_obj_count = 0;
    std::map<std::string, long long> frequency0;
    std::map<std::string, long long> frequency1;
};

std::vector<std::string> single_words(std::string& sentence)
{
    std::vector<std::string> word_vector;
    std::string result_word;
    for (char& character: sentence) {
        if (character== ' ' && result_word.size() != 0) {
            word_vector.push_back(result_word);
            result_word = "";
        }
        else {
            result_word += character;
        }
    }
    word_vector.push_back(result_word);
    return word_vector;
}

void BayesClassifier::preprocessing(std::vector<std::pair<std::string, int>>& data)
{
    this->training_size = data.size();
    for (int i = 0; i < data.size(); ++i) {
        if (data[i].second == 0) {
            ++c0_obj_count;
            std::vector<std::string> words = single_words(data[i].first);
            for (std::string word: words) {
                std::transform(word.begin(), word.end(), word.begin(), tolower);
                ++frequency0[word];
            }
        } else {
            ++c1_obj_count;
            std::vector<std::string> words = single_words(data[i].first);
            for (std::string word: words) {
                std::transform(word.begin(), word.end(), word.begin(), tolower);
                ++frequency1[word];
            }
        }
    }
}

void BayesClassifier::fit(std::vector<std::pair<std::string, int>>& data)
{
    preprocessing(data);
}

int BayesClassifier::predict(std::string& text)
{
    int objects_number = this->training_size;
    long double probabilityC0 = this->c0_obj_count / (long double)objects_number;
    long double probabilityC1 = this->c1_obj_count / (long double)objects_number;
    std::vector<std::string> words = single_words(text);
    long double sample_probabilityc0 = 1.0;
    long double sample_probabilityc1 = 1.0;
    for (std::string word: words) {
        if (frequency0[word] != 0) {
            sample_probabilityc0 *= frequency0[word] / (long double)c0_obj_count;
        } else {
            sample_probabilityc0 *= 0.0001 / (long double)c0_obj_count;
        }
        if (frequency1[word] != 0) {
            sample_probabilityc1 *= frequency1[word] / (long double)c1_obj_count;
        } else {
            sample_probabilityc1 *= 0.0001 / (long double)c1_obj_count;
        }
    }
    if (probabilityC0 * sample_probabilityc0 > probabilityC1 * sample_probabilityc1) {
        return 0;
    } else {
        return 1;
    }
}

void BayesClassifier::predict(std::vector<std::pair<std::string, int>>& data)
{
    this->answers = data.size();
    for (auto object: data) {
        int answer = predict(object.first);
        int true_answer = object.second;
        if (answer == true_answer) ++this->right_answers;
    }
}

double BayesClassifier::getAccurancy()
{
    return this->right_answers / (long double)this->answers;
}

int main()
{
    int n, m;
    BayesClassifier bayes;
    std::vector<std::pair<std::string, int>> data;
    std::cin >> n >> m;
    for (int i = 0; i < m; ++i) {
        std::string c;
        int Class;
        std::cin >> Class;
        std::string text;
        std::getline(std::cin, text);
        std::getline(std::cin, text);
        data.push_back({text, Class});
    }
    bayes.fit(data);
    for (int i = 0; i < n; ++i) {
        std::string text;
        std::getline(std::cin, text);
        std::cout << bayes.predict(text) << "\n";
        std::cout << std::flush;
    }
    return 0;
}