#include <iostream>
#include <string>

std::string Recognizer(cv::Mat input_mat, std::vector<cv::Point2f> pose_result, std::string part_name)
{
        // TODO: 1. Detect face
        //       2. Detect landmark points
        //       3. Recognize expression
        //       4. Return result
        return "none";
}

void test()
{
        std::string lefthand_result =Recognizer(input_mat, pose_result, "left-hand");
        std::string righthand_result =Recognizer(input_mat, pose_result, "right-hand");
        std::string lefteye_result =Recognizer(input_mat, pose_result, "left-eye");
        std::string righteye_result =Recognizer(input_mat, pose_result, "right-eye");
        std::string mouth_result =Recognizer(input_mat, pose_result, "mouth");
}

void main()
{
        test();
}


// 创建单例类
class Singleton {
public:
    // 获取实例方法
    static Singleton* getInstance() {
        if (instance == nullptr) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (instance == nullptr) {
                instance = new Singleton();
            }
        }
        return instance;
    }

private:
    // 构造函数和析构函数为私有
    Singleton() {}
    ~Singleton() {}

    // 禁止拷贝和赋值
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

    // 单例实例指针和互斥锁
    static Singleton* instance;
    static std::mutex mutex_;
};


