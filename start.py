"""
══════════════════════════════════════════════════
  📚 我的知识助手 - 一键启动
══════════════════════════════════════════════════

使用方法:
  1. 双击这个文件 (或在终端输入: python start.py)
  2. 等待提示 "启动成功"
  3. 打开浏览器访问 http://localhost:8000

══════════════════════════════════════════════════
"""
import subprocess
import sys
import os
import time

# 切换到项目目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def ensure_mysql_running():
    """确保 MySQL 在运行，如果没有就启动它"""
    try:
        import pymysql
        conn = pymysql.connect(host="localhost", port=3306, user="rag_user",
                                password="rag_password_2024", database="rag_knowledge_base")
        conn.close()
        return True
    except Exception:
        pass

    # MySQL 没有运行，尝试启动
    mysql_bin = r"C:\mysql\mysql8\bin\mysqld"
    if not os.path.exists(mysql_bin):
        mysql_bin = r"C:\mysql\mysql8\bin\mysqld.exe"

    if os.path.exists(mysql_bin):
        print("🔄 正在启动 MySQL...")
        subprocess.Popen(
            [mysql_bin, "--console"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # 等待启动
        for _ in range(10):
            time.sleep(1.5)
            try:
                import pymysql
                conn = pymysql.connect(host="localhost", port=3306, user="rag_user",
                                        password="rag_password_2024", database="rag_knowledge_base")
                conn.close()
                print("   ✅ MySQL 启动成功！")
                return True
            except Exception:
                pass
        print("   ⚠️ MySQL 启动失败，请手动启动")
        return False
    else:
        print("   ⚠️ 找不到 MySQL，请先安装")
        return False


def main():
    print("═" * 50)
    print("  📚 我的知识助手")
    print("═" * 50)
    print()

    # 1. 检查 Python
    print("🔍 检查环境...")
    py_ver = sys.version_info
    if py_ver < (3, 9):
        print("❌ 需要 Python 3.9 或更高版本！")
        print(f"   当前版本: {py_ver.major}.{py_ver.minor}")
        input("按回车退出...")
        return
    print(f"   ✅ Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}")

    # 2. 检查 MySQL
    print("🗄️  检查 MySQL...")
    mysql_ok = ensure_mysql_running()
    if mysql_ok:
        print("   ✅ MySQL 已就绪")
    else:
        print("   ⚠️ MySQL 不可用，将使用 SQLite 作为备用")
        os.environ["USE_SQLITE_FALLBACK"] = "1"

    # 3. 检查依赖
    print("📦 检查依赖包...")
    missing = []
    for pkg in ["fastapi", "uvicorn", "sqlalchemy", "pymysql",
                 "sentence_transformers", "chromadb"]:
        try:
            __import__(pkg)
            print(f"   ✅ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg} - 需要安装")
            missing.append(pkg)

    if missing:
        print()
        print("⚠️  检测到缺少必要的包，正在自动安装...")
        for pkg in ["fastapi", "uvicorn[standard]", "sqlalchemy", "pymysql",
                     "sentence-transformers", "chromadb", "pymupdf",
                     "pdfminer.six", "python-docx", "openai", "tiktoken",
                     "python-multipart", "aiofiles"]:
            print(f"   📥 安装 {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                           capture_output=True)
        print("✅ 依赖安装完成！\n")

    # 4. LLM 提示
    if not os.getenv("LLM_API_KEY"):
        print("─" * 50)
        print("💡 未设置 LLM_API_KEY，将使用「本地搜索模式」")
        print("   设置 API Key 可启用 AI 智能回答:")
        print("   set LLM_API_KEY=你的Key")
        print("─" * 50)
        print()

    # 5. 启动
    print("🚀 正在启动知识助手...")
    print("   浏览器访问: http://localhost:8000")
    print("   按 Ctrl+C 停止")
    print("=" * 50)
    print()

    try:
        import uvicorn
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 知识助手已关闭，下次见！")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        input("按回车退出...")


if __name__ == "__main__":
    main()
