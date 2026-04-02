module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      // Metro (non-ESM) doesn't support import.meta syntax. Replace it with a
      // safe object so Zustand's dev-only checks don't throw a SyntaxError
      // on web: "Cannot use 'import.meta' outside a module".
      function importMetaPlugin() {
        return {
          visitor: {
            MetaProperty(path) {
              if (
                path.node.meta.name === 'import' &&
                path.node.property.name === 'meta'
              ) {
                path.replaceWithSourceString('({ env: { MODE: "production" } })');
              }
            },
          },
        };
      },
    ],
  };
};
