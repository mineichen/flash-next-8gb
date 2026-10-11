{
  description = "Deterministic dev shell";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };
  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      perSystem = { system, pkgs, ... }:
        let
          commonBuildInputs = [
	    pkgs.git
	    pkgs.stdenv.cc
          ];
          greet = ''
            echo "===================================="
            echo " Welcome to the deterministic dev shell! "
            echo "===================================="
          '';
          policy = pkgs.writeText "policy.json" ''{"default":[{"type":"insecureAcceptAnything"}]}'';
          containername = "qwen-isolated-dev";
          podmanRuntimePath = pkgs.lib.makeBinPath [
            pkgs.slirp4netns
          ];
          podmanRun = "${pkgs.podman}/bin/podman run --rm -it "
            + "--network=slirp4netns "
            + "--tmpfs /tmp "
            + "-v .:/workspace/qwen:z "
            + "-e HOME=/root "
            + "${containername}:latest /bin/entrypoint.sh";
        in
        {
          devShells.default = pkgs.mkShell({
            buildInputs = commonBuildInputs;
            shellHook = greet;
          });
          packages.isolated-build = pkgs.dockerTools.buildImage {
            name = containername;
            tag = "latest";
            copyToRoot = pkgs.buildEnv {
              name = containername;
              paths = commonBuildInputs ++ [
                pkgs.bashInteractive
                pkgs.ripgrep
                pkgs.git
                pkgs.opencode
                pkgs.busybox
                (pkgs.writeScriptBin "entrypoint.sh" ''
                  #!${pkgs.bashInteractive}/bin/bash
                  ${greet}
                  exec ${pkgs.bashInteractive}/bin/bash
                '')
              ];
              pathsToLink = [ "/bin" "/lib" "/include" "/share" ];
            };
            config = {
              Env = [ "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt" "HOME=/root" ];
              Cmd = [ "/bin/entrypoint.sh" ];
              WorkingDir = "/workspace/qwen";
            };
          };
          apps.isolated-build = {
            type = "app";
            program = toString (pkgs.writeShellScript containername ''
              export PATH="${podmanRuntimePath}:$PATH"
              ${pkgs.podman}/bin/podman rmi ${containername} || true
              ${pkgs.podman}/bin/podman load \
                --signature-policy ${policy} \
                --input ${inputs.self.packages.${system}.isolated-build}
              ${podmanRun}
            '');
          };
          apps.isolated-nobuild = {
            type = "app";
            program = toString (pkgs.writeShellScript "run-isolated" ''
              set -euo pipefail
              export PATH="${podmanRuntimePath}:$PATH"
              #if ! ${pkgs.podman}/bin/podman image exists ${containername}:latest 2>/dev/null; then
              #  echo "Image ${containername}:latest not found."
              #  echo "Please build and load it first with:"
              #  echo "  nix run .#isolated-build"
              #  exit 1
              #fi
              ${podmanRun}
            '');
          };
        };
    };
}
