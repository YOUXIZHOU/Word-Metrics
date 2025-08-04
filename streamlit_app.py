import streamlit as st
import pandas as pd
import numpy as np
import base64
import ast

st.set_page_config(page_title="Classifier Word Metrics", layout="wide")
st.title("📊 Classifier Word Metrics")
st.markdown("Transform binary classifier results into word-based metrics at the statement or ID level.")

# --- Upload CSV ---
st.header("1. Upload Your Data")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {uploaded_file.name} with {len(df)} rows.")

        # Column selections
        id_column = st.selectbox("Select ID Column", options=df.columns, index=0)
        text_column = st.selectbox("Select Text/Statement Column", options=df.columns, index=1)
        classifier_columns = st.multiselect(
            "Select Classifier Columns",
            options=[col for col in df.columns if col not in [id_column, text_column]]
        )

        process_mode = st.radio("Processing Mode", ["Statement-level", "Aggregate to ID-level"])

        if st.button("🚀 Process Data"):
            with st.spinner("Processing... Please wait."):
                results = []

                if process_mode == "Statement-level":
                    for idx, row in df.iterrows():
                        statement_text = str(row[text_column])
                        total_word_count = len(statement_text.split())

                        result = {
                            "row_id": idx + 1,
                            "id": row[id_column],
                            "statement": statement_text,
                            "word_count": total_word_count
                        }

                        if "number_likes" in df.columns:
                            result["number_likes"] = row.get("number_likes", 0)
                        if "number_comments" in df.columns:
                            result["number_comments"] = row.get("number_comments", 0)

                        for col in classifier_columns:
                            val = float(row.get(col, 0))
                            result[col] = val

                            base_col = col.removeprefix("has_")
                            found_terms_col = f"found_{base_col}_terms"
                            found_terms = row.get(found_terms_col, [])

                            if isinstance(found_terms, str):
                                try:
                                    parsed = ast.literal_eval(found_terms)
                                    found_terms = parsed if isinstance(parsed, list) else []
                                except:
                                    found_terms = []
                            elif not isinstance(found_terms, list):
                                found_terms = []

                            found_word_count = len(found_terms)
                            percentage = found_word_count / total_word_count if total_word_count > 0 else 0
                            result[f"{col}_percentage"] = percentage * 100

                        results.append(result)

                else:  # ID-level aggregation
                    grouped = df.groupby(id_column)
                    for uid, group in grouped:
                        statements = group[text_column].astype(str).tolist()
                        total_words = sum(len(s.split()) for s in statements)
                        agg_result = {
                            "id": uid,
                            "total_word_count": total_words
                        }

                        if "number_likes" in df.columns:
                            agg_result["total_likes"] = group["number_likes"].sum()
                        if "number_comments" in df.columns:
                            agg_result["total_comments"] = group["number_comments"].sum()

                        for col in classifier_columns:
                            values = group[col].astype(float)
                            base_col = col.removeprefix("has_")
                            found_terms_col = f"found_{base_col}_terms"

                            if found_terms_col in group.columns:
                                positive_rows = group[values > 0]
                                found_counts = []

                                for item in positive_rows[found_terms_col]:
                                    if isinstance(item, str):
                                        try:
                                            parsed = ast.literal_eval(item)
                                            terms = parsed if isinstance(parsed, list) else []
                                        except:
                                            terms = []
                                    elif isinstance(item, list):
                                        terms = item
                                    else:
                                        terms = []
                                    found_counts.append(len(terms))

                                word_count = sum(found_counts)
                            else:
                                word_count = 0

                            positive_ratio = (values > 0).sum() / len(values)
                            agg_result[f"{col}_word_count"] = word_count
                            agg_result[f"{col}_percentage"] = positive_ratio * 100
                            agg_result[f"{col}_continuous_score"] = round(positive_ratio, 3)

                            if "number_likes" in df.columns:
                                agg_result[f"{col}_likes"] = group[values > 0]["number_likes"].sum()
                            if "number_comments" in df.columns:
                                agg_result[f"{col}_comments"] = group[values > 0]["number_comments"].sum()

                        results.append(agg_result)

                result_df = pd.DataFrame(results)
                st.success(f"Processed {len(result_df)} rows.")

                # --- Tactic Impact Summary ---
                if "number_likes" in df.columns and "number_comments" in df.columns and process_mode == "Statement-level":
                    tactic_stats = []
                    for col in classifier_columns:
                        pos_rows = result_df[result_df[col] > 0]
                        grouped = pos_rows.groupby("id")

                        likes_per_post = grouped["number_likes"].first()
                        comments_per_post = grouped["number_comments"].first()

                        total_likes = likes_per_post.sum()
                        avg_likes = likes_per_post.mean()
                        total_comments = comments_per_post.sum()
                        avg_comments = comments_per_post.mean()

                        tactic_stats.append({
                            "tactic": col,
                            "positive_statements": len(pos_rows),
                            "unique_posts": len(grouped),
                            "total_likes": int(total_likes),
                            "avg_likes": round(avg_likes, 2) if not np.isnan(avg_likes) else 0,
                            "total_comments": int(total_comments),
                            "avg_comments": round(avg_comments, 2) if not np.isnan(avg_comments) else 0
                        })

                    summary_df = pd.DataFrame(tactic_stats)
                    st.subheader("🧮 Tactic Impact Summary (Per Unique Post, Based on Positive Statements)")
                    st.dataframe(summary_df, use_container_width=True)

                # --- Preview Results ---
                with st.expander("📊 Preview Results (First 100 Rows)"):
                    st.dataframe(result_df.head(100), use_container_width=True)

                # --- Download CSV ---
                csv = result_df.to_csv(index=False).encode('utf-8')
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="processed_results.csv">📥 Download Full Results CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to read file: {e}")
else:
    st.info("Please upload a CSV file with text and classifier columns.")
