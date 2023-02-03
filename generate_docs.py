import glob
import os

import git
from mlem.api import load
from tqdm import tqdm

repo_path = "data/mlem-docs/mlem.ai"
branch = "nano-gpt"


def checkout():
    if not os.path.exists(repo_path):
        repo = git.Repo.clone_from("https://github.com/iterative/mlem.ai/",
                                   repo_path)
    else:
        repo = git.Repo(repo_path)
        # repo.remote("origin").pull()
    if branch not in [h.name for h in repo.heads]:
        head = repo.create_head(branch)
    else:
        head = [h for h in repo.heads if h.name == branch][0]
    head.checkout()


def main():
    checkout()
    docs_model = load("mlem_char")

    for filename in tqdm(
            glob.glob(repo_path + "/content/**/*.md", recursive=True)):
        relpath = os.path.relpath(filename, repo_path)
        print(relpath)
        start = f"{relpath}\n"
        generated = docs_model(start, temperature=0.3, max_new_tokens=1000)
        with open(filename, "w") as docfile:
            docfile.write(generated[len(start):])


if __name__ == '__main__':
    main()
