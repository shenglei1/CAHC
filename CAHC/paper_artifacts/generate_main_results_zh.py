"""中文绘图入口；与英文版共用数据与统计计算。"""
import generate_main_results as implementation

def main():
    implementation.LANGUAGE = "zh"
    implementation.OUTPUT_DIR = implementation.SCRIPT_DIR / "tables_zh"
    implementation.main()

if __name__ == "__main__":
    main()
