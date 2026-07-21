from dotenv import load_dotenv

load_dotenv()


def main():
    from graph.graph import app

    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    result = app.invoke({"question": "What are the types of agent memory?"})
    print(result["generation"])


if __name__ == "__main__":
    main()
