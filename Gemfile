source "https://rubygems.org"

# Pinned so a Ruby upgrade cannot silently change the build. CI uses
# ruby/setup-ruby with `ruby-version: "3.3"`; declaring the same requirement
# here makes a mismatch fail immediately instead of producing a site built
# with different gems than the committed Gemfile.lock was resolved against.
ruby "~> 3.3"

# The site is built by GitHub Actions and pushed to the `gh-pages` branch,
# so we build with a current Jekyll rather than the pinned GitHub Pages gem.
gem "jekyll", "~> 4.3"

# Enables GitHub-flavoured markdown (fenced code blocks with ```) via
# `kramdown: input: GFM` in _config.yml.
gem "kramdown-parser-gfm", "~> 1.1"

# Serves the site on http://127.0.0.1:4000 while editing.
# Note: jekyll-feed is deliberately absent. The site has no posts, so an
# RSS feed would only produce a 404.

# Windows and JRuby do not include zoneinfo files.
install_if -> { RUBY_PLATFORM =~ %r!mingw|mswin|java! } do
  gem "tzinfo", "~> 2.0"
  gem "tzinfo-data"
end

# Performance booster for watching directories on Windows
gem "wdm", "~> 0.1", :install_if => Gem.win_platform?