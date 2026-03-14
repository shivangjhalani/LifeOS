{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  # dotenv.enable = true;
  dotenv.disableHint = true;

  # https://devenv.sh/basics/
  env.GREET = "LifeOS";

  # https://devenv.sh/packages/
  packages = [
    # pkgs.zlib
  ];

  # https://devenv.sh/languages/
  languages.javascript = {
    npm.enable = true;
    enable = true;
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
    ndk.enable = false;
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
  enterShell = ''
    hello         # Run scripts directly
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
