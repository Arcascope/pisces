import subprocess

def get_git_commit_hash():
    """Return the full commit SHA of the current Git HEAD."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        return commit_hash.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None

if __name__ == "__main__":
    print(get_git_commit_hash())

