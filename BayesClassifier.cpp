#include "BayesClassifier.h"

std::vector<long double> BayesClassifier::getRecall()
{
    long double recall0 = this->class_right_answers[0] / (long double)this->all_class_answers[0];
    long double recall1 = this->class_right_answers[1] / (long double)this->all_class_answers[1];
    long double recall2 = this->class_right_answers[2] / (long double)this->all_class_answers[2];
    std::vector<long double> recalls = {recall0, recall1, recall2};
    return recalls;
}

std::vector<long double> BayesClassifier::getPrecision()
{
    long double precision0 = this->class_right_answers[0] / (long double)this->class_predicted_answers[0];
    long double precision1 = this->class_right_answers[1] / (long double)this->class_predicted_answers[1];
    long double precision2 = this->class_right_answers[2] / (long double)this->class_predicted_answers[2];
    std::vector<long double> precisions = {precision0, precision1, precision2};
    return precisions;
}

std::vector<long double> BayesClassifier::getF1Measure()
{
    std::vector<long double> recalls = getRecall();
    std::vector<long double> precisions = getPrecision();
    long double f1measure0 = 2 * recalls[0] * precisions[0] / (recalls[0] + precisions[0]);
    long double f1measure1 = 2 * recalls[1] * precisions[1] / (recalls[1] + precisions[1]);
    long double f1measure2 = 2 * recalls[2] * precisions[2] / (recalls[2] + precisions[2]);
    std::vector<long double> measures = {f1measure0, f1measure1, f1measure2};
    return measures;
}

std::vector<int> BayesClassifier::GetClass(std::vector<std::string>& ClassNames)
{
    std::vector<int> classes;
    for (auto ClassName: ClassNames) {
        if (ClassName == "Cats") classes.push_back(0);
        else if (ClassName == "Dogs") classes.push_back(1);
        else if (ClassName == "Hamsters") classes.push_back(2);
    }
    return classes;
}

std::vector<std::string> BayesClassifier::GetClass(std::vector<int>& ClassNames)
{
    std::vector<std::string> classes;
    for (auto ClassName: ClassNames) {
        if (ClassName == 0) classes.push_back("Cats");
        else if (ClassName == 1) classes.push_back("Dogs");
        else if (ClassName == 2) classes.push_back("Hamsters");
    }
    return classes;
}

std::vector<std::string> single_words(std::string& sentence);

void BayesClassifier::get_data(const std::string& filename)
{
    std::ifstream input(filename);
    int lines_count;
    std::vector<std::pair<std::string, std::vector<int>>> data;
    while (input >> lines_count) {
        std::string text = "";
        std::string sample_tags, header, sample_line;
        getline(input, sample_tags);
        getline(input, sample_tags);
        getline(input, header);
        for (int i = 0; i < lines_count; ++i) {
            getline(input, sample_line);
            text += sample_line;
            if (i != lines_count - 1) text += " ";
        }
        std::replace(sample_tags.begin(), sample_tags.end(), ',', ' ');
        std::vector<std::string> tags = single_words(sample_tags);
        std::vector<int> classes = GetClass(tags);
        data.push_back({text, classes});
    }
    training_size = data.size();
    fit(data);
}

std::string delete_space(std::string& str)
{
    std::string new_string;
    for (int i = 0; i < str.size(); ++i) {
        if (str[i] != ' ') new_string += str[i];
    }
    return new_string;
}

void BayesClassifier::transform(std::string& word)
{
    std::transform(word.begin(), word.end(), word.begin(), tolower);
    std::replace(word.begin(), word.end(), ',', ' ');
    std::replace(word.begin(), word.end(), '.', ' ');
    std::replace(word.begin(), word.end(), '?', ' ');
    std::replace(word.begin(), word.end(), '`', ' ');
    std::replace(word.begin(), word.end(), ':', ' ');
    std::replace(word.begin(), word.end(), '"', ' ');
    std::replace(word.begin(), word.end(), '!', ' ');
    std::replace(word.begin(), word.end(), '(', ' ');
    std::replace(word.begin(), word.end(), ')', ' ');
    word = delete_space(word);
}

void BayesClassifier::fit(std::vector<std::pair<std::string, std::vector<int>>>& data)
{
    for (auto sample: data) {
        for (auto Class: sample.second) {
            ++obj_count[Class];
        }
        std::vector<std::string> words = single_words(sample.first);
        for (auto word: words) {
            transform(word);
            for (auto Class: sample.second) {
                ++word_frequency[Class][word];
            }
        }
    }
    std::ofstream output(this->output_file);
    output << this->training_size << "\n";
    for (int i = 0; i < CLASS_COUNT; ++i) {
        output << "Class: " << i << " " << obj_count[i] << "\n";
        for (auto object: word_frequency[i]) {
            output << object.first << " " << object.second << "\n";
        }
    }
}

BayesClassifier::BayesClassifier(int argc, char** argv): obj_count(CLASS_COUNT), word_frequency(CLASS_COUNT), 
class_right_answers(CLASS_COUNT), all_class_answers(CLASS_COUNT), class_predicted_answers(CLASS_COUNT)
{
    std::cout << argv[0] << " " << argv[1] << "\n";
    if (argc < 2) {
        std::cout << "too little arguments\n";
        exit(EXIT_FAILURE);
    }
    std::string mode = argv[1];
    if (mode == "learn") {
        if (argc != 6) {
            std::cout << "Incorrect number of arguments\n";
            exit(EXIT_FAILURE);
        } else {
            std::string input_arg = argv[2];
            std::string output_arg = argv[4];
            if (input_arg != "--input") {std::cout << "Incorrect arguments"; exit(EXIT_FAILURE);}
            if (output_arg != "--output") {std::cout << "Incorrect arguments"; exit(EXIT_FAILURE);}
            std::string input_file = argv[3];
            std::string output_file = argv[5];
            this->output_file = output_file;
            get_data(input_file);
        }
    } else if (mode == "classify") {
        if (argc != 8) {
            std::cout << "Incorrect number of arguments\n";
        } else {
            std::string stats_arg = argv[2];
            std::string input_arg = argv[4];
            std::string output_arg = argv[6];
            if (stats_arg != "--stats") {std::cout << "Incorrect arguments"; exit(EXIT_FAILURE);}
            if (input_arg != "--input") {std::cout << "Incorrect arguments"; exit(EXIT_FAILURE);}
            if (output_arg != "--output") {std::cout << "Incorrect arguments"; exit(EXIT_FAILURE);}
            std::string stats_file = argv[3];
            std::string input_file = argv[5];
            std::string output_file = argv[7];
            predict(stats_file, input_file, output_file);
        }
    }
}

void BayesClassifier::stats_preprocessing(std::string& stats_file)
{
    std::ifstream stats(stats_file);
    stats >> training_size;
    std::string line;
    getline(stats, line);
    int class_number;
    int obj_count_;
    int word_count;
    while (getline(stats, line)) {
        if (line == "") break;
        std::string marker;
        std::string new_line;
        std::istringstream is(line);
        is >> marker;
        if (marker == "Class:") {
            is >> class_number; 
            is >> obj_count_;
            this->obj_count[class_number] = obj_count_;
        } else {
            is >> word_count;
            this->word_frequency[class_number][marker] = word_count;
            std::cout << "word: " << marker << " count: " << word_count << "\n";
        }
    }
    stats.close();
}

std::vector<std::pair<std::string, std::string>> BayesClassifier::input_preprocessing(std::string& input_file)
{
    std::ifstream input(input_file);
    int lines_count;
    std::vector<std::pair<std::string, std::string>> data;
    while (input >> lines_count) {
        std::string text = "";
        std::string header, sample_line, sample_tags;
        getline(input, sample_tags);
        getline(input, sample_tags);
        getline(input, header);
        for (int i = 0; i < lines_count; ++i) {
            getline(input, sample_line);
            text += sample_line;
            if (i != lines_count - 1) text += " ";
        }
        std::replace(sample_tags.begin(), sample_tags.end(), ',', ' ');
        std::vector<std::string> tags = single_words(sample_tags);
        std::vector<int> classes = GetClass(tags);
        std::sort(classes.begin(), classes.end());
        this->test_answers[text] = classes;
        data.push_back({text, header});
    }
    this->classify_size = data.size();
    input.close();
    return data;
}

std::map<std::string, std::vector<long double>> BayesClassifier::softmax(std::map<std::string, std::vector<long double>>& probs)
{
    std::map<std::string, std::vector<long double>> predictions;
    for (auto sample: probs) {
        long double probc0 = sample.second[0];
        long double probc1 = sample.second[1];
        long double probc2 = sample.second[2];
        long double res0 = 1 / (long double)(1 + exp(probc1 - probc0) + exp(probc2 - probc0));
        long double res1 = 1 / (long double)(1 + exp(probc0 - probc1) + exp(probc2 - probc1));
        long double res2 = 1 / (long double)(1 + exp(probc1 - probc2) + exp(probc0 - probc2));
        predictions[sample.first].push_back(res0);
        predictions[sample.first].push_back(res1);
        predictions[sample.first].push_back(res2);
    }
    return predictions;
}

void BayesClassifier::predict(std::string& stats_file, std::string& input_file, std::string& output_file)
{
    stats_preprocessing(stats_file);
    std::vector<std::pair<std::string, std::string>> dataset = input_preprocessing(input_file);
    std::map<std::string, std::string> label_text;
    for (auto object: dataset) {
        label_text[object.first] = object.second;
    }
    std::ofstream output(output_file);
    std::map<std::string, std::vector<int>> output_set;
    std::map<std::string, int> unique;
    std::map<std::string, std::vector<long double>> probs;
    for (int i = 0; i < CLASS_COUNT; ++i) {
        for (auto word: word_frequency[i]) {
            ++unique[word.first];
        }
    }
    int unique_count = unique.size();
    for (auto sample: dataset) {
        for (int i = 0; i < CLASS_COUNT; ++i) {
            int word_count = 0;
            for (auto word: word_frequency[i]) word_count += word.second;
            long double ClassProbability;
            ClassProbability = (obj_count[i] + 1) / ((long double)training_size + 3);
            long double ClassProbabilityLn = std::log(ClassProbability);
            std::vector<std::string> words = single_words(sample.first);
            long double sample_probability = 1.0;
            long double sample_probabilityLn = 0;
            for (auto word: words) {
                transform(word);
                sample_probabilityLn += std::log((word_frequency[i][word] + 1) / (long double)(unique_count + word_count));
            }
            probs[sample.first].push_back(ClassProbabilityLn + sample_probabilityLn);
        }
    }
    std::map<std::string, std::vector<long double>> predictions = softmax(probs);
    int curr_obj = 0;
    int cur_tag = 0;
    for (auto sample: this->test_answers) {
        for (auto answer: sample.second) {
            if (answer == 0) ++all_class_answers[0];
            else if (answer == 1) ++all_class_answers[1];
            else if (answer == 2) ++all_class_answers[2];
        }
    }
    for (auto sample: predictions) {
        std::vector<int> predicted_tags;
        for (int i = 0; i < sample.second.size(); ++i) {
            if (sample.second[i] > class_belong_rate) {
                predicted_tags.push_back(i);
            }
        }
        std::sort(predicted_tags.begin(), predicted_tags.end());
        output << "label\n" << label_text[sample.first] << "\n";
        std::vector<std::string> tags = GetClass(predicted_tags);
        output << "tags\n";
        for  (auto tag: tags) {
            output << tag << " ";
        }
        output << "\n";
        for (auto prediction: predicted_tags) {
            if (prediction == 0) ++this->class_predicted_answers[0];
            if (prediction == 1) ++this->class_predicted_answers[1];
            if (prediction == 2) ++this->class_predicted_answers[2];
        }
        int curr_tag = 0;
        int real_tags_count = this->test_answers[sample.first].size();
        for (int i = 0; i < real_tags_count; ++i) {
            if (curr_tag < predicted_tags.size() && predicted_tags[curr_tag] == this->test_answers[sample.first][i]) {
                if (predicted_tags[curr_tag] == 0) ++this->class_right_answers[0];
                else if (predicted_tags[curr_tag] == 1) ++this->class_right_answers[1];
                else if (predicted_tags[curr_tag] == 2) ++this->class_right_answers[2];
                this->right_tag_answers += 1./ real_tags_count;
                ++curr_tag;
            }
        }
        ++curr_obj;
    }
    long double recall0 = this->class_right_answers[0] / (long double)this->all_class_answers[0];
    long double recall1 = this->class_right_answers[1] / (long double)this->all_class_answers[1];
    long double recall2 = this->class_right_answers[2] / (long double)this->all_class_answers[2];
    long double precision0 = this->class_right_answers[0] / (long double)this->class_predicted_answers[0];
    long double precision1 = this->class_right_answers[1] / (long double)this->class_predicted_answers[1];
    long double precision2 = this->class_right_answers[2] / (long double)this->class_predicted_answers[2];
    std::cout << "accurancy: " << this->right_tag_answers / this->classify_size << "\n";
    std::cout << "recall 0: " << this->class_right_answers[0] / (long double)this->all_class_answers[0] << "\n";
    std::cout << "recall 1: " << this->class_right_answers[1] / (long double)this->all_class_answers[1] << "\n";
    std::cout << "recall 2: " << this->class_right_answers[2] / (long double)this->all_class_answers[2] << "\n";
    std::cout << "precision 0: " << this->class_right_answers[0] / (long double)this->class_predicted_answers[0] << "\n";
    std::cout << "precision 1: " << this->class_right_answers[1] / (long double)this->class_predicted_answers[1] << "\n";
    std::cout << "precision 2: " << this->class_right_answers[2] / (long double)this->class_predicted_answers[2] << "\n";
    std::cout << "f1-measure0: " << 2 * recall0 * precision0 / (recall0 + precision0) << "\n";
    std::cout << "f1-measure1: " << 2 * recall1 * precision1 / (recall1 + precision1) << "\n";
    std::cout << "f1-measure2: " << 2 * recall2 * precision2 / (recall2 + precision2) << "\n";
}

std::vector<std::string> single_words(std::string& sentence)
{
    std::vector<std::string> word_vector;
    std::string result_word;
    for (char& character: sentence) {
        if (character == ' ' && result_word.size() != 0) {
            word_vector.push_back(result_word);
            result_word = "";
        }
        else {
            result_word += character;
        }
    }
    word_vector.push_back(result_word);
    std::vector<std::string> word_vector_final;
    for (auto str: word_vector) {
        std::string final_word;
        for (char& c: str) {
            if (c != ' ') final_word += c;
        }
        word_vector_final.push_back(final_word);
    }
    return word_vector_final;
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
        std::cout << "answer: " << answer << " " << "right answer: " << true_answer << "\n";
        if (answer == true_answer) ++this->right_answers;
    }
}

double BayesClassifier::getAccurancy()
{
    std::cout << "accurancy: " << this->right_answers / (long double)this->answers << "\n";
    return this->right_answers / (long double)this->answers;
}