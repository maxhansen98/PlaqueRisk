import os
import hashlib
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-f1", "--file1", required=True,)
    parser.add_argument("-f2", "--file2", required=True,)
    parser.add_argument("-o", "--out", default="/home/dnanexus/tmp")
    parser.add_argument("--anonymize", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # --- Load bulk presence matrix (with IDs) ---
    df = pd.read_csv(args.file1)
    id_col = df.columns[0]
    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col)
    df.index.name = "eid"

    # ensure boolean dtype
    for c in df.columns:
        if df[c].dtype != bool:
            df[c] = df[c].map(lambda x: str(x).strip().lower() in {"true", "1", "t", "yes", "y"})

    prot = pd.read_csv(args.file2)

    prot_ids = prot["eid"].dropna().astype(int).astype(str).unique()
    prot_id_set = set(prot_ids)


    all_ids = sorted(set(df.index).union(prot_id_set))
    df = df.reindex(all_ids).fillna(False)
    df["Proteomics"] = df.index.to_series().isin(prot_id_set).values

    out_path = os.path.join(args.out, "NOEXPORT_20260121_ukb_bulk_merge.csv")
    df.to_csv(out_path, index=True)

    print(f"Wrote merged file with IDs to: {out_path}. Don't forget to upload your results.")
    print(f"Participants: {df.shape[0]}")
    print(f"Proteomics TRUE: {int(df['Proteomics'].sum())}")

    if args.anonymize:
        if args.anonymize:
            salt = os.environ.get("UKB_ANON_SALT")
            if not salt:
                raise RuntimeError(
                    "UKB_ANON_SALT is not set. Set it before running, e.g.\n"
                    "export UKB_ANON_SALT=$(openssl rand -hex 32)"
                )


            def anon_id(eid: str) -> str:
                h = hashlib.sha256((salt + eid).encode("utf-8")).hexdigest()[:16]
                return f"P_{h}"


            df_anon = df.copy()
            df_anon.insert(0, "anon_id", [anon_id(str(x)) for x in df_anon.index.astype(str)])
            df_anon = df_anon.set_index("anon_id", drop=True)
            df_anon.index.name = "anon_id"

            out_path_anon = os.path.join(args.out, "20260121_ukb_bulk_with_Proteomics.csv")
            df_anon.to_csv(out_path_anon, index=True)
            print(f"Wrote anonymized file to: {out_path_anon}. Don't forget to upload your results.")