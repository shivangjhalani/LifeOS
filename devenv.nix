{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  dotenv.enable = true;
  dotenv.disableHint = true;

  # https://devenv.sh/basics/
  env.GREET = "LifeOS";

  # https://devenv.sh/packages/
  packages = [
    pkgs.nodejs_22
    pkgs.rsync
  ];

  # https://devenv.sh/languages/
  languages.javascript = {
    npm.enable = true;
    enable = true;
    package = pkgs.nodejs_22;
    bun = {
      enable = true;
      package = pkgs.bun;
    };
    # Enable Language Server
    lsp.enable = true;
  };

  languages.typescript = {
    enable = true;
    lsp.enable = true;
  };

  android = {
    enable = true;
    reactNative.enable = true;

    # Minimal SDK components
    platforms = {
      version = [ "36" ]; # Android 16 (stable target)
    };

    buildTools = {
      version = [ "36.0.0" ];
    };

    emulator.enable = true;
    systemImages.enable = true;
    # NDK must be included so Gradle doesn't try to install it into the read-only Nix store.
    ndk = {
      enable = true;
      # version = [ "27.1.12297006" ];
    };
    sources.enable = false;
    googleAPIs.enable = false;
  };

  # env.LD_LIBRARY_PATH = lib.makeLibraryPath [
  #   pkgs.stdenv.cc.cc.lib
  #   pkgs.zlib
  # ];

  # https://devenv.sh/processes/
  # processes.dev.exec = "${lib.getExe pkgs.watchexec} -n -- ls -la";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  # https://devenv.sh/basics/
  # Copy Nix Android SDK to writable dir so Gradle can install NDK components (expo-sqlite etc).
  # The Nix store is read-only; Gradle fails with "SDK directory is not writable" otherwise.
  enterShell = ''
    hello
    if [ -n "''${ANDROID_HOME:-}" ] && [ ! -w "$ANDROID_HOME" ]; then
      WRITABLE_SDK="''${XDG_DATA_HOME:-$HOME/.local/share}/android-sdk-lifeos"
      SOURCE_MARKER="$WRITABLE_SDK/.nix-source"
      if [ ! -f "$SOURCE_MARKER" ] || [ "$(cat "$SOURCE_MARKER")" != "$ANDROID_HOME" ]; then
        echo "Copying Android SDK to writable dir (one-time, ~2min)..."
        mkdir -p "$WRITABLE_SDK"
        rsync -a --info=progress2 "$ANDROID_HOME/" "$WRITABLE_SDK/" 2>/dev/null || cp -a "$ANDROID_HOME/"* "$WRITABLE_SDK/"
        chmod -R u+w "$WRITABLE_SDK"
        echo "$ANDROID_HOME" > "$SOURCE_MARKER"
      fi
      export ANDROID_HOME="$WRITABLE_SDK"
      export ANDROID_SDK_ROOT="$WRITABLE_SDK"
    fi
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  # enterTest = ''
  #   echo "Running tests"
  #   git --version | grep --color=auto "${pkgs.git.version}"
  # '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
