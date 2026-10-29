# Links between posts, added when the site is built, so that posts exported from the course repository can be
# exported again without losing them:
#
# - in a lesson of a course on /teaching/ (_data/courses.yml), the first mention of each other published lesson,
#   "lesson 3" or "Lesson 3", links to that lesson;
# - in any post, the first mention of a term from _data/internal_links.yml links to the post that covers it,
#   unless the post is that post or already links to it.
#
# Only running text is touched: never links, code, headings, citations, tables, quotations, captions or the
# reference list. A lesson's own line of links to its neighbours (p.tfs-neighbours) is dropped: the course
# navigation after every lesson (_includes/course_nav.liquid) replaces it.
require "nokogiri"
require "set"

module InternalLinks
  SKIP_ELEMENTS = %w[a code pre h1 h2 h3 h4 h5 h6 sup table blockquote figcaption script style button summary].freeze
  SKIP_CLASSES = %w[tfs-links tfs-refs read-next].freeze
  MAX_TERM_LINKS = 6 # per post, so that a post doesn't turn into a list of links

  def self.course_lessons(site)
    courses = {}
    (site.data["courses"] || []).each do |course|
      lessons = {}
      (course["lessons"] || []).each_with_index do |lesson, k|
        lessons[k + 1] = lesson["url"] if lesson["url"]
      end
      courses[course["id"]] = lessons
    end
    courses
  end

  def self.skipped?(node)
    node.ancestors.any? do |element|
      next false unless element.element?
      SKIP_ELEMENTS.include?(element.name) || (element["class"].to_s.split & SKIP_CLASSES).any?
    end
  end

  def self.link_first(fragment, pattern, url)
    fragment.xpath(".//text()").each do |node|
      next if skipped?(node)
      match = pattern.match(node.content)
      next unless match
      document = fragment.document
      link = Nokogiri::XML::Node.new("a", document)
      link["href"] = url
      link.content = match[0]
      # the text node keeps what comes before the match; the link and the rest follow it. (Adding a text node
      # next to another one makes libxml2 merge the two, so nothing is inserted before the node.)
      node.content = match.pre_match
      node.add_next_sibling(link)
      link.add_next_sibling(Nokogiri::XML::Text.new(match.post_match, document)) unless match.post_match.empty?
      return true
    end
    false
  end

  def self.linked_urls(fragment)
    fragment.css("a[href]").map { |a| a["href"].sub(%r{^https?://xmarva\.github\.io}, "").sub(/#.*/, "") }.to_set
  end

  def self.add(post)
    site = post.site
    own = post.url
    fragment = Nokogiri::HTML5::DocumentFragment.parse(post.content)
    changed = false
    courses = course_lessons(site).values.select { |lessons| lessons.value?(own) }

    # a lesson's own line of links to its neighbours: _includes/course_nav.liquid shows them for every lesson
    unless courses.empty?
      fragment.css("p.tfs-neighbours").each do |line|
        line.remove
        changed = true
      end
    end
    present = linked_urls(fragment)

    courses.each do |lessons|
      lessons.each do |number, url|
        next if url == own || present.include?(url)
        if link_first(fragment, /\b[Ll]esson #{number}\b(?![.,]?\d)/, url)
          present << url
          changed = true
        end
      end
    end

    added = 0
    (site.data["internal_links"] || []).each do |rule|
      break if added >= MAX_TERM_LINKS
      url = rule["url"]
      next if url == own || present.include?(url)
      next if (rule["not_in"] || []).include?(own)
      pattern = Regexp.new(rule["pattern"], rule["case_sensitive"] ? nil : Regexp::IGNORECASE)
      if link_first(fragment, pattern, url)
        present << url
        added += 1
        changed = true
      end
    end

    post.content = fragment.to_html if changed
  end
end

Jekyll::Hooks.register :posts, :post_convert do |post|
  InternalLinks.add(post) unless post.data["internal_links"] == false
end
