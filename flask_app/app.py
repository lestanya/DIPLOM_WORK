from flask import Flask, render_template, request, redirect, url_for, Response
import pandas as pd
import plotly.express as px

from db_utils import (
    init_db,
    get_all_requests,
    delete_request,
    update_request,
    update_request_status,
    get_statistics,
    export_to_csv_bytes,
)

app = Flask(__name__)
init_db()


@app.route("/")
def index():
    return redirect(url_for("requests_page"))


@app.route("/requests")
def requests_page():
    sort_by = request.args.get("sort_by", "timestamp")
    order = request.args.get("order", "desc")
    rows = get_all_requests(sort_by=sort_by, order=order)
    return render_template("request.html", rows=rows, sort_by=sort_by, order=order)


@app.route("/requests/delete/<int:request_id>", methods=["POST"])
def remove_request(request_id):
    delete_request(request_id)
    return redirect(url_for("requests_page"))


@app.route("/requests/update/<int:request_id>", methods=["POST"])
def edit_request(request_id):
    data = {
        "text": request.form.get("text"),
        "address": request.form.get("address"),
        "category": request.form.get("category"),
        "emotion": request.form.get("emotion"),
        "urgency": request.form.get("urgency"),
        "name": request.form.get("name"),
        "phone": request.form.get("phone"),
    }
    update_request(request_id, data)
    return redirect(url_for("requests_page"))


@app.route("/requests/status/<int:request_id>", methods=["POST"])
def change_status(request_id):
    new_status = request.form.get("status")
    update_request_status(request_id, new_status)
    return redirect(url_for("requests_page"))


@app.route("/statistics")
def statistics_page():
    stats = get_statistics()

    df_cat = pd.DataFrame(list(stats["categories"].items()), columns=["category", "count"])
    df_urg = pd.DataFrame(list(stats["urgencies"].items()), columns=["urgency", "count"])
    df_status = pd.DataFrame(list(stats["statuses"].items()), columns=["status", "count"])
    df_emo = pd.DataFrame(list(stats["emotions"].items()), columns=["emotion", "count"])

    fig_cat = px.pie(df_cat, names="category", values="count", title="Категории обращений") if not df_cat.empty else None
    fig_urg = px.bar(df_urg, x="urgency", y="count", title="Срочность") if not df_urg.empty else None
    fig_status = px.bar(df_status, x="status", y="count", title="Статусы") if not df_status.empty else None
    fig_emo = px.pie(df_emo, names="emotion", values="count", title="Эмоции") if not df_emo.empty else None

    graphs = {
        "cat": fig_cat.to_html(full_html=False, include_plotlyjs="cdn") if fig_cat else "<p>Нет данных</p>",
        "urg": fig_urg.to_html(full_html=False, include_plotlyjs=False) if fig_urg else "<p>Нет данных</p>",
        "status": fig_status.to_html(full_html=False, include_plotlyjs=False) if fig_status else "<p>Нет данных</p>",
        "emo": fig_emo.to_html(full_html=False, include_plotlyjs=False) if fig_emo else "<p>Нет данных</p>",
    }

    return render_template("statistics.html", graphs=graphs)


@app.route("/csv")
def csv_page():
    return render_template("csv_load.html")


@app.route("/csv/download")
def csv_download():
    csv_text = export_to_csv_bytes()
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=complaints.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True)