"""中文绘图入口；与英文版共用数据与统计计算。"""
import generate_additional_figures as implementation

def main():
    implementation.LANGUAGE = "zh"
    implementation.main()

if __name__ == "__main__":
    main()
