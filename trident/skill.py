from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path

from .global_utils import TRIDENT_GLOBALS


GITHUB_SKILL_URL = "https://raw.githubusercontent.com/wgobrien/trident/main/SKILL.md"
FALLBACK_SKILL_URL = "https://web.cs.ucla.edu/~wob/projects/trident/SKILL.md"

ROOT_REQUIRED_FLAGS = ["--reads", "--library", "--output", "--database"]
ROOT_OPTIONAL_FLAGS = [
    "--pair",
    "--pathogen-dict",
    "--threads",
    "--save-figs",
    "--low-memory",
    "--seed",
    "--device",
    "--num-latent",
    "--concentration",
    "--batch-size",
    "--steps",
    "--learning-rate",
    "--num-samples",
    "--keep-top",
    "--spaces",
    "--k",
    "--l",
    "--max-var",
    "--continue-training",
    "--rebuild",
    "--help",
]
CHECK_REQUIRED_FLAGS = ["--reads", "--library", "--output"]
CHECK_OPTIONAL_FLAGS = [
    "--pair",
    "--pathogen-dict",
    "--threads",
    "--save-figs",
    "--low-memory",
    "--seed",
    "--device",
    "--help",
]
SKILL_INSTALL_REQUIRED_FLAGS = ["--dest"]
SKILL_INSTALL_OPTIONAL_FLAGS = ["--source", "--help"]
SKILL_SHOW_FLAGS = ["--dest", "--help"]
SKILL_PATH_FLAGS = ["--dest", "--help"]
SKILL_AUTOCOMPLETE_FLAGS = ["--shell", "--dest", "--help"]

ROOT_REQUIRED_DESCS = [
    ("--reads", "[required] reads file"),
    ("--library", "[required] reference library"),
    ("--output", "[required] output directory"),
    ("--database", "[required] Kraken2 database"),
]
ROOT_OPTIONAL_DESCS = [
    ("--pair", "paired-end reads file"),
    ("--pathogen-dict", "custom pathogen_dict.json"),
    ("--threads", "number of threads"),
    ("--save-figs", "save diagnostic figures"),
    ("--low-memory", "avoid SQLite db construction"),
    ("--seed", "random seed"),
    ("--device", "compute device"),
    ("--num-latent", "number of latent factors"),
    ("--concentration", "coverage beta prior concentration"),
    ("--batch-size", "SVI subsample size"),
    ("--steps", "SVI iterations"),
    ("--learning-rate", "SVI learning rate"),
    ("--num-samples", "posterior sample count"),
    ("--keep-top", "taxid filter quantile"),
    ("--spaces", "spaced seed masking"),
    ("--k", "kmer size"),
    ("--l", "minimizer window length"),
    ("--max-var", "maximum taxa/accessions"),
    ("--continue-training", "resume previous training"),
    ("--rebuild", "rebuild data from scratch"),
    ("--help", "show help"),
]
CHECK_REQUIRED_DESCS = [
    ("--reads", "[required] reads file"),
    ("--library", "[required] reference library"),
    ("--output", "[required] output directory"),
]
CHECK_OPTIONAL_DESCS = [
    ("--pair", "paired-end reads file"),
    ("--pathogen-dict", "custom pathogen_dict.json"),
    ("--threads", "number of threads"),
    ("--save-figs", "save diagnostic figures"),
    ("--low-memory", "avoid SQLite db construction"),
    ("--seed", "random seed"),
    ("--device", "compute device"),
    ("--help", "show help"),
]
SKILL_INSTALL_DESCS = [
    ("--dest", "[required] destination directory or markdown path"),
    ("--source", "custom local source path"),
    ("--help", "show help"),
]
SKILL_SHOW_DESCS = [
    ("--dest", "optional: show a specific installed skill"),
    ("--help", "show help"),
]
SKILL_PATH_DESCS = [
    ("--dest", "optional: resolve a specific destination"),
    ("--help", "show help"),
]
SKILL_AUTOCOMPLETE_DESCS = [
    ("--shell", "shell type: bash or zsh"),
    ("--dest", "output file for completion script"),
    ("--help", "show help"),
]


def skill_target(dest: str) -> Path:
    path = Path(dest).expanduser()
    if path.suffix.lower() == ".md":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return (path / "SKILL.md").resolve()


def require_dest(args) -> str:
    if not getattr(args, "dest", None):
        raise ValueError("`--dest` is required.")
    return args.dest


def registry_path() -> Path:
    save_folder = Path(TRIDENT_GLOBALS["save_folder"]).expanduser()
    save_folder.mkdir(parents=True, exist_ok=True)
    return save_folder / "skill_installs.json"


def load_registry() -> list:
    path = registry_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    installs = data.get("installs", [])
    return [str(item) for item in installs]


def save_registry(installs: list) -> None:
    path = registry_path()
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"installs": installs}, handle, indent=2)


def record_install(path: Path) -> None:
    installs = [entry for entry in load_registry() if entry != str(path)]
    installs.append(str(path))
    save_registry(installs)


def skill_show_command(args) -> None:
    if getattr(args, "dest", None):
        path = skill_target(args.dest)
    else:
        installs = load_registry()
        if not installs:
            raise RuntimeError("No saved skill install paths found in {}".format(registry_path()))
        path = Path(installs[-1])

    if not path.exists():
        raise FileNotFoundError("Could not find installed SKILL.md at {}".format(path))
    print(path.read_text(encoding="utf-8"), end="")


def skill_path_command(args) -> None:
    if getattr(args, "dest", None):
        print(str(skill_target(args.dest)))
        return

    installs = load_registry()
    if not installs:
        raise RuntimeError("No saved skill install paths found in {}".format(registry_path()))
    print(installs[-1])


def skill_install_command(args) -> None:
    target = skill_target(require_dest(args))

    if args.source:
        source = Path(args.source).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError("Custom source does not exist: {}".format(source))
        shutil.copyfile(str(source), str(target))
        record_install(target)
        print("Installed {} from {}".format(target, source))
        return

    errors = []
    for url in (GITHUB_SKILL_URL, FALLBACK_SKILL_URL):
        try:
            with urllib.request.urlopen(url) as response:
                content = response.read().decode("utf-8")
            target.write_text(content, encoding="utf-8")
            record_install(target)
            print("Installed {} from {}".format(target, url))
            return
        except Exception as exc:
            errors.append("{}: {}".format(url, exc))

    raise RuntimeError(
        "Failed to install SKILL.md from all remote sources:\n{}".format("\n".join(errors))
    )


def _join_flags(flags: list) -> str:
    return " ".join(flags)


def _bash_completion_script() -> str:
    root_flags = _join_flags(ROOT_REQUIRED_FLAGS + ROOT_OPTIONAL_FLAGS)
    accession_flags = _join_flags(ROOT_REQUIRED_FLAGS + ROOT_OPTIONAL_FLAGS)
    check_flags = _join_flags(CHECK_REQUIRED_FLAGS + CHECK_OPTIONAL_FLAGS)
    skill_install_flags = _join_flags(SKILL_INSTALL_REQUIRED_FLAGS + SKILL_INSTALL_OPTIONAL_FLAGS)
    skill_show_flags = _join_flags(SKILL_SHOW_FLAGS)
    skill_path_flags = _join_flags(SKILL_PATH_FLAGS)
    skill_autocomplete_flags = _join_flags(SKILL_AUTOCOMPLETE_FLAGS)

    return """# bash completion for trident
_trident_completion() {
    local cur prev cmd1 cmd2 opts
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev=""
    if [ "${COMP_CWORD}" -ge 1 ]; then
        prev="${COMP_WORDS[COMP_CWORD-1]}"
    fi
    cmd1=""
    cmd2=""
    if [ "${#COMP_WORDS[@]}" -ge 2 ]; then
        cmd1="${COMP_WORDS[1]}"
    fi
    if [ "${#COMP_WORDS[@]}" -ge 3 ]; then
        cmd2="${COMP_WORDS[2]}"
    fi

    if type compopt >/dev/null 2>&1; then
        compopt -o nosort 2>/dev/null
    fi

    case "${prev}" in
        --source)
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        --dest)
            COMPREPLY=( $(compgen -f -d -- "${cur}") )
            return 0
            ;;
        --shell)
            COMPREPLY=( $(compgen -W "bash zsh" -- "${cur}") )
            return 0
            ;;
    esac

    if [ "${cur}" = "-" ] || [ "${cur#-}" != "${cur}" ]; then
        case "${cmd1}" in
            autocomplete)
                opts='__SKILL_AUTOCOMPLETE_FLAGS__'
                ;;
            accession)
                opts='__ACCESSION_FLAGS__'
                ;;
            check)
                opts='__CHECK_FLAGS__'
                ;;
            skill)
                case "${cmd2}" in
                    show)
                        opts='__SKILL_SHOW_FLAGS__'
                        ;;
                    path)
                        opts='__SKILL_PATH_FLAGS__'
                        ;;
                    *)
                        opts='__SKILL_INSTALL_FLAGS__'
                        ;;
                esac
                ;;
            *)
                opts='__ROOT_FLAGS__'
                ;;
        esac
        COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
        return 0
    fi

    case "${cmd1}" in
        "")
            COMPREPLY=( $(compgen -W "accession check skill autocomplete __ROOT_FLAGS__" -- "${cur}") )
            ;;
        skill)
            if [ "${COMP_CWORD}" -eq 2 ]; then
                COMPREPLY=( $(compgen -W "show path __SKILL_INSTALL_FLAGS__" -- "${cur}") )
            else
                case "${cmd2}" in
                    show)
                        COMPREPLY=( $(compgen -W "__SKILL_SHOW_FLAGS__" -- "${cur}") )
                        ;;
                    path)
                        COMPREPLY=( $(compgen -W "__SKILL_PATH_FLAGS__" -- "${cur}") )
                        ;;
                    *)
                        COMPREPLY=( $(compgen -W "__SKILL_INSTALL_FLAGS__" -- "${cur}") )
                        ;;
                esac
            fi
            ;;
        autocomplete)
            COMPREPLY=( $(compgen -W "__SKILL_AUTOCOMPLETE_FLAGS__" -- "${cur}") )
            ;;
        accession)
            COMPREPLY=( $(compgen -W "__ACCESSION_FLAGS__" -- "${cur}") )
            ;;
        check)
            COMPREPLY=( $(compgen -W "__CHECK_FLAGS__" -- "${cur}") )
            ;;
        *)
            COMPREPLY=( $(compgen -W "__ROOT_FLAGS__" -- "${cur}") )
            ;;
    esac
    return 0
}
complete -F _trident_completion trident
""".replace("__ROOT_FLAGS__", root_flags).replace(
        "__ACCESSION_FLAGS__", accession_flags
    ).replace(
        "__CHECK_FLAGS__", check_flags
    ).replace(
        "__SKILL_INSTALL_FLAGS__", skill_install_flags
    ).replace(
        "__SKILL_SHOW_FLAGS__", skill_show_flags
    ).replace(
        "__SKILL_PATH_FLAGS__", skill_path_flags
    ).replace(
        "__SKILL_AUTOCOMPLETE_FLAGS__", skill_autocomplete_flags
    )


def _zsh_completion_script() -> str:
    root_required_flags = " ".join(ROOT_REQUIRED_FLAGS)
    root_optional_flags = " ".join(ROOT_OPTIONAL_FLAGS)
    check_required_flags = " ".join(CHECK_REQUIRED_FLAGS)
    check_optional_flags = " ".join(CHECK_OPTIONAL_FLAGS)
    skill_install_required_flags = " ".join(SKILL_INSTALL_REQUIRED_FLAGS)
    skill_install_optional_flags = " ".join(SKILL_INSTALL_OPTIONAL_FLAGS)
    skill_show_flags = " ".join(SKILL_SHOW_FLAGS)
    skill_path_flags = " ".join(SKILL_PATH_FLAGS)
    skill_autocomplete_flags = " ".join(SKILL_AUTOCOMPLETE_FLAGS)

    return """#compdef trident

compdef _trident trident

_trident() {
  local -a root_cmds skill_cmds
  local prev
  prev="${words[CURRENT-1]}"
    root_cmds=(accession check skill autocomplete)
    skill_cmds=(show path)

  case "${prev}" in
    --reads|--pair|--library|--source|--pathogen-dict)
      _files
      return
      ;;
    --output|--database)
      _files -/
      return
      ;;
    --dest)
      _files
      return
      ;;
    --shell)
      compadd -Q -- bash zsh
      return
      ;;
  esac

  _trident_flag_groups() {
    local required_name="$1"
    local optional_name="$2"
    local -a required_flags optional_flags required_labels
    local flag
    required_flags=("${(@P)required_name}")
    optional_flags=("${(@P)optional_name}")
    required_labels=()
    for flag in "${required_flags[@]}"; do
      required_labels+=("${flag}[required]")
    done
    _wanted required-options expl '[required]' compadd -Q -d required_labels -- "${required_flags[@]}"
    _wanted optional-options expl 'optional options' compadd -Q -- "${optional_flags[@]}"
  }

  if (( CURRENT == 2 )); then
        _wanted commands expl 'commands' compadd -Q -- "${root_cmds[@]}"
        local -a root_required_flags root_optional_flags
        root_required_flags=(__ROOT_REQUIRED_FLAGS__)
        root_optional_flags=(__ROOT_OPTIONAL_FLAGS__)
        _trident_flag_groups root_required_flags root_optional_flags
        return
  fi

  case "$words[2]" in
    accession)
      local -a accession_required_flags accession_optional_flags
      accession_required_flags=(__ROOT_REQUIRED_FLAGS__)
      accession_optional_flags=(__ROOT_OPTIONAL_FLAGS__)
      _trident_flag_groups accession_required_flags accession_optional_flags
      ;;
    autocomplete)
      local -a skill_autocomplete_required_flags skill_autocomplete_optional_flags
      skill_autocomplete_required_flags=()
      skill_autocomplete_optional_flags=(__SKILL_AUTOCOMPLETE_FLAGS__)
      _trident_flag_groups skill_autocomplete_required_flags skill_autocomplete_optional_flags
      ;;
    check)
      local -a check_required_flags check_optional_flags
      check_required_flags=(__CHECK_REQUIRED_FLAGS__)
      check_optional_flags=(__CHECK_OPTIONAL_FLAGS__)
      _trident_flag_groups check_required_flags check_optional_flags
      ;;
    skill)
      if (( CURRENT == 3 )); then
        _wanted skill-commands expl 'skill commands' compadd -Q -- "${skill_cmds[@]}"
        local -a skill_install_required_flags skill_install_optional_flags
        skill_install_required_flags=(__SKILL_INSTALL_REQUIRED_FLAGS__)
        skill_install_optional_flags=(__SKILL_INSTALL_OPTIONAL_FLAGS__)
        _trident_flag_groups skill_install_required_flags skill_install_optional_flags
        return
      fi
      case "$words[3]" in
        show)
          local -a skill_show_flags
          skill_show_flags=(__SKILL_SHOW_FLAGS__)
          _wanted options expl 'options' compadd -Q -- "${skill_show_flags[@]}"
          ;;
        path)
          local -a skill_path_required_flags skill_path_optional_flags
          skill_path_required_flags=()
          skill_path_optional_flags=(__SKILL_PATH_FLAGS__)
          _trident_flag_groups skill_path_required_flags skill_path_optional_flags
          ;;
        *)
          local -a skill_install_required_flags skill_install_optional_flags
          skill_install_required_flags=(__SKILL_INSTALL_REQUIRED_FLAGS__)
          skill_install_optional_flags=(__SKILL_INSTALL_OPTIONAL_FLAGS__)
          _trident_flag_groups skill_install_required_flags skill_install_optional_flags
          ;;
      esac
      ;;
    *)
      local -a root_required_flags root_optional_flags
      root_required_flags=(__ROOT_REQUIRED_FLAGS__)
      root_optional_flags=(__ROOT_OPTIONAL_FLAGS__)
      _trident_flag_groups root_required_flags root_optional_flags
      ;;
  esac
}
""".replace("__ROOT_REQUIRED_FLAGS__", root_required_flags).replace(
        "__ROOT_OPTIONAL_FLAGS__", root_optional_flags
    ).replace(
        "__CHECK_REQUIRED_FLAGS__", check_required_flags
    ).replace(
        "__CHECK_OPTIONAL_FLAGS__", check_optional_flags
    ).replace(
        "__SKILL_INSTALL_REQUIRED_FLAGS__", skill_install_required_flags
    ).replace(
        "__SKILL_INSTALL_OPTIONAL_FLAGS__", skill_install_optional_flags
    ).replace(
        "__SKILL_SHOW_FLAGS__", skill_show_flags
    ).replace(
        "__SKILL_PATH_FLAGS__", skill_path_flags
    ).replace(
        "__SKILL_AUTOCOMPLETE_FLAGS__", skill_autocomplete_flags
    )


def detect_shell(args) -> str:
    shell = getattr(args, "shell", None)
    if shell:
        return shell
    shell_env = os.environ.get("SHELL", "")
    shell_name = os.path.basename(shell_env)
    if shell_name in ("bash", "zsh"):
        return shell_name
    return "bash"


def autocomplete_target(args, shell: str) -> Path:
    dest = getattr(args, "dest", None)
    if dest:
        path = Path(dest).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    if shell == "zsh":
        path = Path("~/.zsh_completion.d/trident.zsh").expanduser()
    else:
        path = Path("~/.bash_completion.d/trident.sh").expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def skill_autocomplete_command(args) -> None:
    shell = detect_shell(args)
    if shell == "bash":
        script = _bash_completion_script()
    elif shell == "zsh":
        script = _zsh_completion_script()
    else:
        raise ValueError("Unsupported shell: {}".format(shell))

    target = autocomplete_target(args, shell)
    target.write_text(script, encoding="utf-8")
    print("[INFO] {} completion file for trident saved to {}".format(shell, target))
    print("Run: source {}".format(target))
